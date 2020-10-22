#include <algorithm>
#include <numeric>

#include <thrust/device_vector.h>

#include <torch/extension.h>

#include <quiver/common.hpp>
#include <quiver/functor.cu.hpp>
#include <quiver/quiver.cu.hpp>
#include <quiver/reindex.cu.hpp>
#include <quiver/trace.hpp>
#include <quiver/zip.hpp>

namespace quiver
{
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
    using W = float;

    // deprecated, not compatible with AliGraph
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    sample_sub(const torch::Tensor &vertices, int k) const
    {
        return sample_sub_with_stream(0, vertices, k);
    }

    std::tuple<torch::Tensor, torch::Tensor>
    sample_once(const torch::Tensor &vertices, int k) const
    {
        TRACE(__func__);

        thrust::device_vector<T> inputs;
        thrust::device_vector<T> outputs;
        thrust::device_vector<T> output_counts;

        return sample_kernel(0, vertices, k, inputs, outputs, output_counts);
    }

    std::tuple<torch::Tensor, torch::Tensor>
    sample_kernel(const cudaStream_t stream, const torch::Tensor &vertices,
                  int k, thrust::device_vector<T> &inputs,
                  thrust::device_vector<T> &outputs,
                  thrust::device_vector<T> &output_counts) const
    {
        T tot = 0;
        const auto policy = thrust::cuda::par.on(stream);
        thrust::device_vector<T> output_ptr;
        thrust::device_vector<T> output_eid;

        check_eq<long>(vertices.dim(), 1);
        const size_t bs = vertices.size(0);

        {
            TRACE("alloc_1");
            inputs.resize(bs);
            output_counts.resize(bs);
            output_ptr.resize(bs);
        }
        // output_ptr is exclusive prefix sum of output_counts(neighbor counts
        // <= k)
        {
            TRACE("prepare");
            thrust::copy(vertices.data_ptr<long>(),
                         vertices.data_ptr<long>() + bs, inputs.begin());
            this->degree(stream, inputs.data(), inputs.data() + inputs.size(),
                         output_counts.data());
            if (k >= 0) {
                thrust::transform(policy, output_counts.begin(),
                                  output_counts.end(), output_counts.begin(),
                                  cap_by<T>(k));
            }
            thrust::exclusive_scan(policy, output_counts.begin(),
                                   output_counts.end(), output_ptr.begin());
            tot = thrust::reduce(policy, output_counts.begin(),
                                 output_counts.end());
        }
        {
            TRACE("alloc_2");
            outputs.resize(tot);
            output_eid.resize(tot);
        }
        // outputs[outptr[i], outptr[i + 1]) are unique neighbors of inputs[i]
        {
            TRACE("sample");
            this->sample(stream, inputs.begin(), inputs.end(),
                         output_ptr.begin(), output_counts.begin(),
                         outputs.data(), output_eid.data());
        }
        torch::Tensor neighbor = torch::empty(tot, vertices.options());
        torch::Tensor eid = torch::empty(tot, vertices.options());
        thrust::copy(outputs.begin(), outputs.end(), neighbor.data_ptr<T>());
        thrust::copy(output_eid.begin(), output_eid.end(), eid.data_ptr<T>());
        return std::make_tuple(neighbor, eid);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    sample_sub_with_stream(const cudaStream_t stream,
                           const torch::Tensor &vertices, int k) const
    {
        TRACE(__func__);
        const auto policy = thrust::cuda::par.on(stream);
        const size_t bs = vertices.size(0);

        thrust::device_vector<T> inputs;
        thrust::device_vector<T> outputs;
        thrust::device_vector<T> output_counts;
        thrust::device_vector<T> subset;

        sample_kernel(stream, vertices, k, inputs, outputs, output_counts);
        T tot = outputs.size();

        // reindex
        {
            {
                TRACE("reindex 0");
                subset.resize(inputs.size() + outputs.size());
                thrust::copy(policy, inputs.begin(), inputs.end(),
                             subset.begin());
                thrust::copy(policy, outputs.begin(), outputs.end(),
                             subset.begin() + inputs.size());
                thrust::sort(policy, subset.begin(), subset.end());
                subset.erase(thrust::unique(subset.begin(), subset.end()),
                             subset.end());
                _reindex_with(policy, outputs, subset, outputs);
            }
            {
                TRACE("permute");
                thrust::device_vector<T> s1;
                s1.reserve(subset.size());
                _reindex_with(policy, inputs, subset, s1);
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
};

TorchQuiver new_quiver_from_edge_index(size_t n,
                                       const torch::Tensor &edge_index,
                                       const torch::Tensor &edge_id)
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
    const T *p_id = edge_id.data_ptr<T>();
    std::vector<T> eid(m);
    thrust::copy(p_id, p_id + m, eid.begin());
    return TorchQuiver((T)n, std::move(ei), std::move(eid));
}

TorchQuiver new_quiver_from_edge_index_weight(size_t n,
                                              const torch::Tensor &edge_index,
                                              const torch::Tensor &edge_id,
                                              const torch::Tensor &edge_weight)
{
    TRACE(__func__);
    using T = typename TorchQuiver::T;
    using W = typename TorchQuiver::W;
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
    const T *p_id = edge_id.data_ptr<T>();
    const W *p_weight = edge_weight.data_ptr<W>();
    std::vector<T> eid(m);
    std::vector<W> weight(m);
    thrust::copy(p_id, p_id + m, eid.begin());
    thrust::copy(p_weight, p_weight + m, weight.begin());
    return TorchQuiver((T)n, std::move(ei), std::move(eid), std::move(weight));
}
}  // namespace quiver

void register_cuda_quiver(pybind11::module &m)
{
    m.def("new_quiver_from_edge_index", &quiver::new_quiver_from_edge_index);
    m.def("new_quiver_from_edge_index_weight",
          &quiver::new_quiver_from_edge_index_weight);
    py::class_<quiver::TorchQuiver>(m, "Quiver")
        .def("sample_sub", &quiver::TorchQuiver::sample_sub)
        .def("sample", &quiver::TorchQuiver::sample_once);
}
