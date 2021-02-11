#include <algorithm>
#include <numeric>

#include <thrust/device_vector.h>

#include <pybind11/numpy.h>
#include <torch/extension.h>

#include <quiver/common.hpp>
#include <quiver/functor.cu.hpp>
#include <quiver/quiver.cu.hpp>
#include <quiver/reindex.cu.hpp>
#include <quiver/stream_pool.hpp>
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

class TorchQuiver
{
    using torch_quiver_t = quiver<int64_t, CUDA>;
    torch_quiver_t quiver_;
    stream_pool pool_;

  public:
    TorchQuiver(torch_quiver_t quiver, int device = 0, int num_workers = 4)
        : quiver_(std::move(quiver))
    {
        pool_ = stream_pool(num_workers);
    }

    using T = int64_t;
    using W = float;

    // deprecated, not compatible with AliGraph
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    sample_sub(const torch::Tensor &vertices, int k) const
    {
        return sample_sub_with_stream(0, vertices, k);
    }

    std::tuple<torch::Tensor, torch::Tensor>
    sample_neighbor(int stream_num, const torch::Tensor &vertices, int k)
    {
        cudaStream_t stream = 0;
        if (!pool_.empty()) { stream = (pool_)[stream_num]; }
        const auto policy = thrust::cuda::par.on(stream);
        const size_t bs = vertices.size(0);
        thrust::device_vector<T> inputs;
        thrust::device_vector<T> outputs;
        thrust::device_vector<T> output_counts;
        sample_kernel(stream, vertices, k, inputs, outputs, output_counts);
        torch::Tensor neighbors =
            torch::empty(outputs.size(), vertices.options());
        torch::Tensor counts =
            torch::empty(vertices.size(0), vertices.options());
        thrust::copy(outputs.begin(), outputs.end(), neighbors.data_ptr<T>());
        thrust::copy(output_counts.begin(), output_counts.end(),
                     counts.data_ptr<T>());
        return std::make_tuple(neighbors, counts);
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
        const T *p = vertices.data_ptr<T>();
        const size_t bs = vertices.size(0);

        {
            TRACE_SCOPE("alloc_1");
            inputs.resize(bs);
            output_counts.resize(bs);
            output_ptr.resize(bs);
        }
        // output_ptr is exclusive prefix sum of output_counts(neighbor counts
        // <= k)
        {
            TRACE_SCOPE("prepare");
            thrust::copy(p, p + bs, inputs.begin());
            // quiver_.to_local(stream, inputs);
            quiver_.degree(stream, inputs.data(), inputs.data() + inputs.size(),
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
            TRACE_SCOPE("alloc_2");
            outputs.resize(tot);
            // output_eid.resize(tot);
        }
        // outputs[outptr[i], outptr[i + 1]) are unique neighbors of inputs[i]
        {
            TRACE_SCOPE("sample");
            quiver_.sample(stream, inputs.begin(), inputs.end(),
                           output_ptr.begin(), output_counts.begin(),
                           outputs.data(), output_eid.data());
        }
        torch::Tensor out_neighbor;
        torch::Tensor out_eid;

        // thrust::copy(outputs.begin(), outputs.end(),
        //              out_neighbor.data_ptr<T>());
        // thrust::copy(output_eid.begin(), output_eid.end(),
        //              out_eid.data_ptr<T>());
        return std::make_tuple(out_neighbor, out_eid);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    reindex_group(int stream_num, torch::Tensor orders, torch::Tensor inputs,
                  torch::Tensor counts, torch::Tensor outputs,
                  torch::Tensor out_counts)
    {
        cudaStream_t stream = 0;
        if (!pool_.empty()) { stream = (pool_)[stream_num]; }
        const auto policy = thrust::cuda::par.on(stream);
        thrust::device_vector<T> total_orders(inputs.size(0));
        thrust::device_vector<T> total_inputs(inputs.size(0));
        thrust::device_vector<T> total_counts(inputs.size(0));
        thrust::device_vector<T> prefix_sum(inputs.size(0));
        thrust::device_vector<T> output_sum(inputs.size(0));
        thrust::device_vector<T> values(outputs.size(0));
        thrust::device_vector<T> total_outputs(outputs.size(0));
        thrust::device_vector<T> output_counts(inputs.size(0));
        const T *ptr;
        int bs;
        ptr = inputs.data_ptr<T>();
        bs = inputs.size(0);
        thrust::copy(ptr, ptr + bs, total_inputs.begin());
        ptr = orders.data_ptr<T>();
        thrust::copy(ptr, ptr + bs, total_orders.begin());
        ptr = counts.data_ptr<T>();
        thrust::copy(ptr, ptr + bs, prefix_sum.begin());
        ptr = out_counts.data_ptr<T>();
        thrust::copy(ptr, ptr + bs, output_counts.begin());
        ptr = outputs.data_ptr<T>();
        bs = outputs.size(0);
        thrust::copy(ptr, ptr + bs, values.begin());
        thrust::exclusive_scan(policy, prefix_sum.begin(), prefix_sum.end(),
                               prefix_sum.begin());
        thrust::exclusive_scan(policy, output_counts.begin(),
                               output_counts.end(), output_sum.begin());
        reorder_output(prefix_sum, output_sum, total_orders, output_counts,
                       values, total_outputs, stream);

        thrust::device_vector<T> subset;
        reindex_kernel(stream, total_inputs, total_outputs, subset);

        int tot = total_outputs.size();
        torch::Tensor out_vertices =
            torch::empty(subset.size(), inputs.options());
        torch::Tensor row_idx = torch::empty(tot, inputs.options());
        torch::Tensor col_idx = torch::empty(tot, inputs.options());
        {
            TRACE_SCOPE("prepare output");
            std::vector<T> counts(total_inputs.size());
            std::vector<T> seq(total_inputs.size());
            thrust::copy(output_counts.begin(), output_counts.end(),
                         counts.begin());
            std::iota(seq.begin(), seq.end(), 0);

            replicate_fill(total_inputs.size(), counts.data(), seq.data(),
                           row_idx.data_ptr<T>());
            thrust::copy(subset.begin(), subset.end(),
                         out_vertices.data_ptr<T>());
            thrust::copy(total_outputs.begin(), total_outputs.end(),
                         col_idx.data_ptr<T>());
        }
        return std::make_tuple(out_vertices, row_idx, col_idx);
    }

    void reindex_kernel(const cudaStream_t stream,
                        thrust::device_vector<T> &inputs,
                        thrust::device_vector<T> &outputs,
                        thrust::device_vector<T> &subset) const
    {
        const auto policy = thrust::cuda::par.on(stream);
        // reindex
        {
            {
                TRACE_SCOPE("reindex 0");
                subset.resize(inputs.size() + outputs.size());
                thrust::copy(policy, inputs.begin(), inputs.end(),
                             subset.begin());
                thrust::copy(policy, outputs.begin(), outputs.end(),
                             subset.begin() + inputs.size());
                thrust::sort(policy, subset.begin(), subset.end());
                subset.erase(
                    thrust::unique(policy, subset.begin(), subset.end()),
                    subset.end());
                _reindex_with(policy, outputs, subset, outputs);
            }
            {
                TRACE_SCOPE("permute");
                thrust::device_vector<T> s1;
                s1.reserve(subset.size());
                _reindex_with(policy, inputs, subset, s1);
                complete_permutation(s1, subset.size(), stream);
                subset = permute(s1, subset, stream);

                thrust::device_vector<T> s2;
                inverse_permutation(s1, s2, stream);
                permute_value(s2, outputs, stream);
            }
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    sample_sub_with_stream(int stream_num, const torch::Tensor &vertices,
                           int k) const
    {
        TRACE_SCOPE(__func__);
        cudaStream_t stream = 0;
        if (!pool_.empty()) { stream = (pool_)[stream_num]; }
        const auto policy = thrust::cuda::par.on(stream);
        thrust::device_vector<T> inputs;
        thrust::device_vector<T> outputs;
        thrust::device_vector<T> output_counts;
        thrust::device_vector<T> subset;
        sample_kernel(stream, vertices, k, inputs, outputs, output_counts);
        int tot = outputs.size();

        reindex_kernel(stream, inputs, outputs, subset);

        torch::Tensor out_vertices =
            torch::empty(subset.size(), vertices.options());
        torch::Tensor row_idx = torch::empty(tot, vertices.options());
        torch::Tensor col_idx = torch::empty(tot, vertices.options());
        {
            TRACE_SCOPE("prepare output");
            std::vector<T> counts(output_counts.size());
            std::vector<T> seq(output_counts.size());
            thrust::copy(output_counts.begin(), output_counts.end(),
                         counts.begin());
            std::iota(seq.begin(), seq.end(), 0);

            replicate_fill(inputs.size(), counts.data(), seq.data(),
                           row_idx.data_ptr<T>());
            thrust::copy(subset.begin(), subset.end(),
                         out_vertices.data_ptr<T>());
            thrust::copy(outputs.begin(), outputs.end(), col_idx.data_ptr<T>());
        }
        return std::make_tuple(out_vertices, row_idx, col_idx);
    }
};

// TODO: remove `n` and reuse code
TorchQuiver new_quiver_from_edge_index(size_t n,  //
                                       py::array_t<int64_t> &input_edges,
                                       py::array_t<int64_t> &input_edge_idx,
                                       int device = 0)
{
    cudaSetDevice(device);
    TRACE_SCOPE(__func__);
    using T = typename TorchQuiver::T;
    py::buffer_info edges = input_edges.request();
    py::buffer_info edge_idx = input_edge_idx.request();
    check_eq<int64_t>(edges.ndim, 2);
    check_eq<int64_t>(edges.shape[0], 2);
    const size_t m = edges.shape[1];
    check_eq<int64_t>(edge_idx.ndim, 1);

    bool use_eid = edge_idx.shape[0] == m;

    thrust::device_vector<T> row_idx(m);
    thrust::device_vector<T> col_idx(m);
    {
        const T *p = reinterpret_cast<const T *>(edges.ptr);
        thrust::copy(p, p + m, row_idx.begin());
        thrust::copy(p + m, p + m * 2, col_idx.begin());
    }
    thrust::device_vector<T> edge_idx_;
    if (use_eid) {
        edge_idx_.resize(m);
        const T *p = reinterpret_cast<const T *>(edge_idx.ptr);
        thrust::copy(p, p + m, edge_idx_.begin());
    }
    using Q = quiver<int64_t, CUDA>;
    Q quiver = Q::New(static_cast<T>(n), std::move(row_idx), std::move(col_idx),
                      std::move(edge_idx_));
    return TorchQuiver(std::move(quiver), device);
}

TorchQuiver
new_quiver_from_edge_index_weight(size_t n, py::array_t<int64_t> &input_edges,
                                  py::array_t<int64_t> &input_edge_idx,
                                  py::array_t<float> &input_edge_weight,
                                  int device = 0)
{
    cudaSetDevice(device);
    TRACE_SCOPE(__func__);
    using T = typename TorchQuiver::T;
    using W = typename TorchQuiver::W;
    py::buffer_info edges = input_edges.request();
    py::buffer_info edge_idx = input_edge_idx.request();
    py::buffer_info edge_weight = input_edge_weight.request();
    check_eq<int64_t>(edges.ndim, 2);
    check_eq<int64_t>(edges.shape[0], 2);
    const size_t m = edges.shape[1];
    check_eq<int64_t>(edge_idx.ndim, 1);
    bool use_eid = edge_idx.shape[0] == m;
    check_eq<int64_t>(edge_weight.ndim, 1);
    check_eq<int64_t>(edge_weight.shape[0], m);

    thrust::device_vector<T> row_idx(m);
    thrust::device_vector<T> col_idx(m);
    {
        const T *p = reinterpret_cast<const T *>(edges.ptr);
        thrust::copy(p, p + m, row_idx.begin());
        thrust::copy(p + m, p + m * 2, col_idx.begin());
    }
    thrust::device_vector<T> edge_idx_;
    if (use_eid) {
        edge_idx_.resize(m);
        const T *p = reinterpret_cast<const T *>(edge_idx.ptr);
        thrust::copy(p, p + m, edge_idx_.begin());
    }
    thrust::device_vector<W> edge_weight_(m);
    {
        const W *p = reinterpret_cast<const W *>(edge_weight.ptr);
        thrust::copy(p, p + m, edge_weight_.begin());
    }
    using Q = quiver<int64_t, CUDA>;
    Q quiver = Q::New(static_cast<T>(n), std::move(row_idx), std::move(col_idx),
                      std::move(edge_idx_), std::move(edge_weight_));
    return TorchQuiver(std::move(quiver), device);
}
}  // namespace quiver

void register_cuda_quiver(pybind11::module &m)
{
    m.def("new_quiver_from_edge_index", &quiver::new_quiver_from_edge_index);
    m.def("new_quiver_from_edge_index_weight",
          &quiver::new_quiver_from_edge_index_weight);
    py::class_<quiver::TorchQuiver>(m, "Quiver")
        .def("sample_sub", &quiver::TorchQuiver::sample_sub_with_stream,
             py::call_guard<py::gil_scoped_release>())
        .def("sample_neighbor", &quiver::TorchQuiver::sample_neighbor,
             py::call_guard<py::gil_scoped_release>())
        .def("reindex_group", &quiver::TorchQuiver::reindex_group,
             py::call_guard<py::gil_scoped_release>());
    py::class_<quiver::stream_pool>(m, "StreamPool").def(py::init<int>());
}
