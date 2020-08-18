#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_map>

#include <thrust/device_vector.h>

#include <torch/extension.h>

#include <quiver/common.hpp>
#include <quiver/quiver.cu.hpp>
#include <quiver/reindex.cu.hpp>
#include <quiver/trace.hpp>

namespace quiver
{
class SparseMatrixCuda
{
  public:
    SparseMatrixCuda(const torch::Tensor &coo) {}
    ~SparseMatrixCuda() {}
};

template <typename T>
SparseMatrixCuda new_sparse_matrix_cuda_(const T *row_idx, const T *col_idx,
                                         size_t n, const torch::Tensor &_)
{
    TRACE(__func__);
    {
        TRACE("thrust::copy");
        thrust::device_vector<T> s(n);
        thrust::device_vector<T> e(n);
        thrust::copy(row_idx, row_idx + n, s.begin());
        thrust::copy(col_idx, col_idx + n, e.begin());
    }
    {
        TRACE("thrust::sort");
        thrust::device_vector<thrust::pair<int, int>> edges(n);
        thrust::sort(edges.begin(), edges.end());
    }
    {
        TRACE("std::sort");
        std::vector<std::pair<T, T>> edges(n);
        for (size_t i = 0; i < n; ++i) {
            edges[i].first = row_idx[i];
            edges[i].second = col_idx[i];
        }
        std::sort(edges.begin(), edges.end());
    }
    SparseMatrixCuda spm(_);
    return spm;
}

SparseMatrixCuda new_sparse_matrix_cuda(const torch::Tensor &coo)
{
    check(coo.is_contiguous());
    check_eq<long>(coo.dim(), 2);
    check_eq<int64_t>(coo.size(0), 2);
    const auto n = coo.size(1);
    const auto dtype = coo.dtype();

    if (dtype.Match<int>()) {
        const int *p = coo.data_ptr<int>();
        return new_sparse_matrix_cuda_<int>(p, p + n, n, coo);
    }
    if (dtype.Match<long>()) {
        const long *p = coo.data_ptr<long>();
        return new_sparse_matrix_cuda_<long>(p, p + n, n, coo);
    }
    throw std::runtime_error(std::string("unsupported type: ") +
                             static_cast<std::string>(dtype.name()));
}

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

    torch::Tensor sample_adj(const torch::Tensor &vertices, int k) const
    {
        check_eq<long>(vertices.dim(), 1);
        const size_t bs = vertices.size(0);
        std::vector<T> output_counts(bs);
        std::vector<T> outputs(bs * k);
        this->sample(vertices.size(0), vertices.data_ptr<T>(), k,
                     output_counts.data(), outputs.data());

        const T tot =
            std::accumulate(output_counts.begin(), output_counts.end(), 0);
        printf("sample[%d] from %d, got %d\n", k, (int)bs, (int)tot);
        torch::Tensor out_vertices = torch::empty(tot, vertices.options());
        compact(bs, (size_t)k, output_counts.data(), outputs.data(),
                out_vertices.data_ptr<T>());
        return out_vertices;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    sample_sub(const torch::Tensor &vertices, int k) const
    {
        check_eq<long>(vertices.dim(), 1);
        const size_t bs = vertices.size(0);
        std::vector<T> output_counts(bs);
        std::vector<T> outputs(bs * k);
        this->sample(bs, vertices.data_ptr<T>(), k, output_counts.data(),
                     outputs.data());

        const T tot =
            std::accumulate(output_counts.begin(), output_counts.end(), 0);

        // if (std::find(output_counts.begin(), output_counts.end(), 0) !=
        //     output_counts.end()) {
        //     printf("!!! output_counts contains zero!\n");
        // }

        torch::Tensor row_idx = torch::empty(tot, vertices.options());
        torch::Tensor col_idx = torch::empty(tot, vertices.options());

        replicate_fill(bs, output_counts.data(), vertices.data_ptr<T>(),
                       row_idx.data_ptr<T>());
        compact(bs, (size_t)k, output_counts.data(), outputs.data(),
                col_idx.data_ptr<T>());

        const auto new_idx =
            reindex_cuda(bs, vertices.data_ptr<T>(), tot, row_idx.data_ptr<T>(),
                         col_idx.data_ptr<T>());
        // {
        //     TRACE("compare cuda");
        //     replicate_fill(bs, output_counts.data(), vertices.data_ptr<T>(),
        //                    row_idx.data_ptr<T>());
        //     compact(bs, (size_t)k, output_counts.data(), outputs.data(),
        //             col_idx.data_ptr<T>());

        //     const auto new_idx2 =
        //         reindex_cuda(bs, vertices.data_ptr<T>(), tot,
        //                      row_idx.data_ptr<T>(), col_idx.data_ptr<T>());
        //     // TODO: assert new_idx2 == new_idx
        // }

        torch::Tensor out_vertices =
            torch::empty(new_idx.size(), vertices.options());
        std::copy(new_idx.begin(), new_idx.end(), out_vertices.data_ptr<T>());

        return std::make_tuple(out_vertices, row_idx, col_idx);
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
    py::class_<quiver::SparseMatrixCuda>(m, "SparseMatrixCuda");
    m.def("new_sparse_matrix_cuda", &quiver::new_sparse_matrix_cuda);

    m.def("new_quiver_from_edge_index", &quiver::new_quiver_from_edge_index);
    py::class_<quiver::TorchQuiver>(m, "Quiver")
        .def("sample_adj", &quiver::TorchQuiver::sample_adj)
        .def("sample_sub", &quiver::TorchQuiver::sample_sub);
}
