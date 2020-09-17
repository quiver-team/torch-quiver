#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_map>

#include <thrust/device_vector.h>

#include <torch/extension.h>

#include <quiver/common.hpp>
#include <quiver/functor.cu.hpp>
#include <quiver/quiver.cu.hpp>
#include <quiver/reindex.cu.hpp>
#include <quiver/trace.hpp>

namespace quiver
{
template <typename T>
size_t compact(size_t n, size_t k, const T *counts, const T *values, T *outputs)
{
    size_t off = 0;
    size_t l = 0;
    for (size_t i = 0; i < n; ++i) {
        const size_t c = counts[i];
        std::copy(values + off, values + off + c, outputs + l);
        off += k;
        l += c;
    }
    return l;
}

template <typename T>
void replicate_fill(size_t n, const T *counts, const T *values, T *outputs)
{
    for (size_t i = 0; i < n; ++i) {
        const size_t c = counts[i];
        std::fill(outputs, outputs + c, values[i]);
        outputs += c;
    }
}

using torch_quiver_t = quiver<int64_t, CUDA>;

class TorchQuiver : public torch_quiver_t
{
    using torch_quiver_t::torch_quiver_t;

  public:
    using T = int64_t;

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    sample_sub(const torch::Tensor &vertices, int k) const
    {
        TRACE(__func__);

        thrust::device_vector<T> inputs;
        thrust::device_vector<T> output_counts;
        thrust::device_vector<T> output_ptr;
        thrust::device_vector<T> outputs;
        thrust::device_vector<T> subset;

        check_eq<long>(vertices.dim(), 1);
        const size_t bs = vertices.size(0);
        {
            TRACE("alloc_1");
            inputs.resize(bs);
            output_counts.resize(bs);
            output_ptr.resize(bs);
        }
        T tot = 0;
        {
            TRACE("prepare");
            thrust::copy(vertices.data_ptr<long>(),
                         vertices.data_ptr<long>() + bs, inputs.begin());
            this->degree(inputs.data(), inputs.data() + inputs.size(),
                         output_counts.data());
            if (k >= 0) {
                thrust::transform(output_counts.begin(), output_counts.end(),
                                  output_counts.begin(), cap_by<T>(k));
            }
            thrust::exclusive_scan(output_counts.begin(), output_counts.end(),
                                   output_ptr.begin());
            tot = thrust::reduce(output_counts.begin(), output_counts.end());
        }
        {
            TRACE("alloc_2");
            outputs.resize(tot);
        }
        {
            TRACE("sample");
            this->sample(inputs.begin(), inputs.end(), output_ptr.begin(),
                         output_counts.begin(), outputs.data());
        }

        // reindex
        {
            {
                TRACE("reindex 0");
                subset.resize(inputs.size() + outputs.size());
                thrust::copy(inputs.begin(), inputs.end(), subset.begin());
                thrust::copy(outputs.begin(), outputs.end(),
                             subset.begin() + inputs.size());
                thrust::sort(subset.begin(), subset.end());
                subset.erase(thrust::unique(subset.begin(), subset.end()),
                             subset.end());
                _reindex_with(outputs, subset, outputs);
            }
            {
                TRACE("permute");
                thrust::device_vector<T> s1;
                s1.reserve(subset.size());
                _reindex_with(inputs, subset, s1);
                complete_permutation(s1, subset.size());
                subset = permute(s1, subset);

                thrust::device_vector<T> s2;
                inverse_permutation(s1, s2);
                permute_value(s2, outputs);
            }

            torch::Tensor out_vertices =
                torch::empty(subset.size(), vertices.options());
            torch::Tensor row_idx = torch::empty(tot, vertices.options());
            torch::Tensor col_idx = torch::empty(tot, vertices.options());
            {
                TRACE("prepare output");
                std::vector<T> counts(output_counts.size());
                std::vector<T> seq(output_counts.size());
                thrust::copy(output_counts.begin(), output_counts.end(),
                             counts.begin());
                std::iota(seq.begin(), seq.end(), 0);

                replicate_fill(bs, counts.data(), seq.data(),
                               row_idx.data_ptr<T>());
                thrust::copy(subset.begin(), subset.end(),
                             out_vertices.data_ptr<T>());
                thrust::copy(outputs.begin(), outputs.end(),
                             col_idx.data_ptr<T>());
            }
            return std::make_tuple(out_vertices, row_idx, col_idx);
        }
    }

  private:
    std::vector<T> reindex_cuda(size_t l, const T *seeds, size_t r, T *row_idx,
                                T *col_idx) const
    {
        TRACE(__func__);
        std::vector<T> a;
        std::vector<T> b;
        std::vector<T> c;
        {
            TRACE("reindex_cuda::alloc");
            a.resize(2 * r);
            b.resize(2 * r);
            c.resize(2 * r);
        }
        {
            TRACE("reindex_cuda::cp1");
            std::copy(row_idx, row_idx + r, a.begin());
            std::copy(col_idx, col_idx + r, a.begin() + r);
        }
        {
            TRACE("reindex_with_seeds");
            reindex_with_seeds(l, seeds, a.size(), a.data(), b, c);
        }
        {
            TRACE("reindex_cuda::cp2");
            std::copy(c.begin(), c.begin() + r, row_idx);
            std::copy(c.begin() + r, c.end(), col_idx);
        }
        return b;
    }

    std::vector<T> reindex(size_t l, const T *seeds, size_t r, T *row_idx,
                           T *col_idx) const
    {
        TRACE(__func__);
        std::vector<T> outputs;
        {
            TRACE("reindex::reserve");
            outputs.reserve(l + r);
        }
        std::unordered_map<T, T> idx;

        const auto get_idx = [&](T v) {
            const auto it = idx.find(v);
            if (it == idx.end()) {
                const T new_idx = outputs.size();
                idx[v] = new_idx;
                outputs.push_back(v);
                return new_idx;
            } else {
                return it->second;
            }
        };
        {
            TRACE("reindex 0");
            for (size_t i = 0; i < l; ++i) { get_idx(seeds[i]); }
        }
        {
            TRACE("reindex 1");
            for (size_t i = 0; i < r; ++i) { row_idx[i] = idx.at(row_idx[i]); }
        }
        {
            TRACE("reindex 2");
            for (size_t i = 0; i < r; ++i) { col_idx[i] = get_idx(col_idx[i]); }
        }
        return outputs;
    }
};

TorchQuiver new_quiver_from_edge_index(size_t n,
                                       const torch::Tensor &edge_index)
{
    TRACE(__func__);
    using T = typename TorchQuiver::T;
    check(edge_index.is_contiguous());
    check_eq<int64_t>(edge_index.dim(), 2);
    check_eq<int64_t>(edge_index.size(0), 2);
    const size_t m = edge_index.size(1);
    const T *p = edge_index.data_ptr<T>();
    using vec = std::vector<std::pair<T, T>>;
    vec ei(m);
    {
        TRACE("zip edge_index");
        zip(p, p + m, p + m, &ei[0].first);
    }
    return TorchQuiver((T)n, std::move(ei));
}
}  // namespace quiver

void register_sparse_matrix_cuda(pybind11::module &m)
{
    m.def("new_quiver_from_edge_index", &quiver::new_quiver_from_edge_index);
    py::class_<quiver::TorchQuiver>(m, "Quiver")
        .def("sample_sub", &quiver::TorchQuiver::sample_sub);
}
