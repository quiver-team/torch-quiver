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

#include <thrust/remove.h>

#include <chrono>
#include <ctime>
using namespace std::chrono;

#define CHECK_CUDA(x)                                                          \
    AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")

template <typename IdType>
HostOrderedHashTable<IdType> *
FillWithDuplicates(const IdType *const input, const size_t num_input,
                   cudaStream_t stream,
                   thrust::device_vector<IdType> &unique_items)
{
    system_clock::time_point tp0 = system_clock::now();
    system_clock::duration d0 = tp0.time_since_epoch();
    time_t m0 = duration_cast<microseconds>(d0).count();

    const auto policy = thrust::cuda::par.on(stream);
    const int64_t num_tiles = (num_input + TILE_SIZE - 1) / TILE_SIZE;

    const dim3 grid(num_tiles);
    const dim3 block(BLOCK_SIZE);

    auto host_table = new HostOrderedHashTable<IdType>(num_input, 1);
    DeviceOrderedHashTable<IdType> device_table = host_table->DeviceHandle();
    system_clock::time_point tp1 = system_clock::now();
    system_clock::duration d1 = tp1.time_since_epoch();
    time_t m1 = duration_cast<microseconds>(d1).count();
    // std::cout << "reindex prepare" << m1 - m0 << std::endl;
    // std::cout << "input size" << num_input << std::endl;
    // std::cout << "grid size" << num_tiles << std::endl;
    // std::cout << "block size" << BLOCK_SIZE << std::endl;

    generate_hashmap_duplicates<IdType, BLOCK_SIZE, TILE_SIZE>
        <<<grid, block, 0, stream>>>(input, num_input, device_table);
    thrust::device_vector<int> item_prefix(num_input + 1, 0);

    system_clock::time_point tp2 = system_clock::now();
    system_clock::duration d2 = tp2.time_since_epoch();
    time_t m2 = duration_cast<microseconds>(d2).count();
    // std::cout << "reindex hash" << m2 - m1 << std::endl;

    using it = thrust::counting_iterator<IdType>;
    using Mapping = typename DeviceOrderedHashTable<IdType>::Mapping;
    thrust::for_each(it(0), it(num_input),
                     [count = thrust::raw_pointer_cast(item_prefix.data()),
                      table = device_table,
                      in = input] __device__(IdType i) mutable {
                         Mapping &mapping = *(table.Search(in[i]));
                         if (mapping.index == i) { count[i] = 1; }
                     });
    thrust::exclusive_scan(item_prefix.begin(), item_prefix.end(),
                           item_prefix.begin());
    size_t tot = item_prefix[num_input];
    // std::cout << "next size" << tot << std::endl;
    unique_items.resize(tot);

    thrust::for_each(it(0), it(num_input),
                     [prefix = thrust::raw_pointer_cast(item_prefix.data()),
                      table = device_table, in = input,
                      u = thrust::raw_pointer_cast(
                          unique_items.data())] __device__(IdType i) mutable {
                         Mapping &mapping = *(table.Search(in[i]));
                         if (mapping.index == i) {
                             mapping.local = prefix[i];
                             u[prefix[i]] = in[i];
                         }
                     });
    system_clock::time_point tp3 = system_clock::now();
    system_clock::duration d3 = tp3.time_since_epoch();
    time_t m3 = duration_cast<microseconds>(d3).count();
    // size_t workspace_bytes;
    // CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    //     nullptr, workspace_bytes, static_cast<IdType *>(nullptr),
    //     static_cast<IdType *>(nullptr), grid.x + 1));
    // void *workspace = device->AllocWorkspace(ctx_, workspace_bytes);

    // CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes,
    //                                         item_prefix, item_prefix,
    //                                         grid.x + 1, stream));
    // device->FreeWorkspace(ctx_, workspace);

    // compact_hashmap<IdType, BLOCK_SIZE, TILE_SIZE><<<grid, block, 0,
    // stream>>>(
    //     input, num_input, device_table, item_prefix, unique, num_unique);
    // CUDA_CALL(cudaGetLastError());
    // device->FreeWorkspace(ctx_, item_prefix);

    // std::cout << "reindex search" << m3 - m2 << std::endl;
    return host_table;
}

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
        thrust::device_vector<T> output_idx;
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
            output_idx.resize(tot);
        }
        // outputs[outptr[i], outptr[i + 1]) are unique neighbors of inputs[i]
        // {
        //     TRACE_SCOPE("sample");
        //     quiver_.sample(stream, inputs.begin(), inputs.end(),
        //                    output_ptr.begin(), output_counts.begin(),
        //                    outputs.data(), output_eid.data());
        // }
        {
            TRACE_SCOPE("sample");
            quiver_.new_sample(
                stream, k, thrust::raw_pointer_cast(inputs.data()),
                inputs.size(), thrust::raw_pointer_cast(output_ptr.data()),
                thrust::raw_pointer_cast(output_counts.data()),
                thrust::raw_pointer_cast(outputs.data()),
                thrust::raw_pointer_cast(output_idx.data()));
        }
        torch::Tensor out_neighbor;
        torch::Tensor out_eid;

        // thrust::copy(outputs.begin(), outputs.end(),
        //              out_neighbor.data_ptr<T>());
        // thrust::copy(output_eid.begin(), output_eid.end(),
        //              out_eid.data_ptr<T>());
        return std::make_tuple(out_neighbor, out_eid);
    }

    static void reindex_kernel(const cudaStream_t stream,
                               thrust::device_vector<T> &inputs,
                               thrust::device_vector<T> &outputs,
                               thrust::device_vector<T> &subset)
    {
        const auto policy = thrust::cuda::par.on(stream);
        HostOrderedHashTable<T> *table;
        // reindex
        {
            {
                TRACE_SCOPE("reindex 0");
                subset.resize(inputs.size() + outputs.size());
                thrust::copy(policy, inputs.begin(), inputs.end(),
                             subset.begin());
                thrust::copy(policy, outputs.begin(), outputs.end(),
                             subset.begin() + inputs.size());
                thrust::device_vector<T> unique_items;
                unique_items.clear();
                table =
                    FillWithDuplicates(thrust::raw_pointer_cast(subset.data()),
                                       subset.size(), stream, unique_items);
                subset.resize(unique_items.size());
                thrust::copy(policy, unique_items.begin(), unique_items.end(),
                             subset.begin());
                // thrust::sort(policy, subset.begin(), subset.end());
                // subset.erase(
                //     thrust::unique(policy, subset.begin(), subset.end()),
                //     subset.end());
                // _reindex_with(policy, outputs, subset, outputs);
            }
            {
                TRACE_SCOPE("permute");
                // thrust::device_vector<T> s1;
                // s1.reserve(subset.size());
                // _reindex_with(policy, inputs, subset, s1);
                // complete_permutation(s1, subset.size(), stream);
                // subset = permute(s1, subset, stream);

                // thrust::device_vector<T> s2;
                // inverse_permutation(s1, s2, stream);
                // permute_value(s2, outputs, stream);
                DeviceOrderedHashTable<T> device_table = table->DeviceHandle();
                thrust::for_each(
                    policy, outputs.begin(), outputs.end(),
                    [device_table] __device__(T & id) mutable {
                        using Iterator =
                            typename DeviceOrderedHashTable<T>::Iterator;
                        Iterator iter = device_table.Search(id);
                        id = static_cast<T>((*iter).local);
                    });
            }
            delete table;
        }
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

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    sample_sub_with_stream(int stream_num, const torch::Tensor &vertices,
                           int k) const
    {
        system_clock::time_point tp0 = system_clock::now();
        system_clock::duration d0 = tp0.time_since_epoch();
        time_t m0 = duration_cast<microseconds>(d0).count();
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
        system_clock::time_point tp1 = system_clock::now();
        system_clock::duration d1 = tp1.time_since_epoch();
        time_t m1 = duration_cast<microseconds>(d1).count();
        // std::cout << "sample" << m1 - m0 << std::endl;

        reindex_kernel(stream, inputs, outputs, subset);
        system_clock::time_point tp2 = system_clock::now();
        system_clock::duration d2 = tp2.time_since_epoch();
        time_t m2 = duration_cast<microseconds>(d2).count();
        // std::cout << "reindex" << m2 - m1 << std::endl;

        torch::Tensor out_vertices =
            torch::empty(subset.size(), vertices.options());
        torch::Tensor row_idx = torch::empty(tot, vertices.options());
        torch::Tensor col_idx = torch::empty(tot, vertices.options());
        {
            TRACE_SCOPE("prepare output");
            thrust::device_vector<T> prefix_count(output_counts.size());
            thrust::device_vector<T> seq(output_counts.size());
            thrust::sequence(policy, seq.begin(), seq.end());
            thrust::exclusive_scan(policy, output_counts.begin(),
                                   output_counts.end(), prefix_count.begin());

            const size_t m = inputs.size();
            using it = thrust::counting_iterator<T>;
            thrust::for_each(
                policy, it(0), it(m),
                [prefix = thrust::raw_pointer_cast(prefix_count.data()),
                 count = thrust::raw_pointer_cast(output_counts.data()),
                 in = thrust::raw_pointer_cast(seq.data()),
                 out = thrust::raw_pointer_cast(
                     row_idx.data_ptr<T>())] __device__(T i) {
                    for (int j = 0; j < count[i]; j++) {
                        out[prefix[i] + j] = in[i];
                    }
                });
            thrust::copy(subset.begin(), subset.end(),
                         out_vertices.data_ptr<T>());
            thrust::copy(outputs.begin(), outputs.end(), col_idx.data_ptr<T>());
        }
        system_clock::time_point tp3 = system_clock::now();
        system_clock::duration d3 = tp3.time_since_epoch();
        time_t m3 = duration_cast<microseconds>(d3).count();
        // std::cout << "output" << m3 - m2 << std::endl;
        return std::make_tuple(out_vertices, row_idx, col_idx);
    }
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
reindex_all(torch::Tensor orders, torch::Tensor inputs, torch::Tensor outputs,
            torch::Tensor count)
{
    using T = int64_t;
    cudaStream_t stream = 0;
    const auto policy = thrust::cuda::par.on(stream);
    thrust::device_vector<T> total_inputs(inputs.size(0));
    thrust::device_vector<T> total_outputs(outputs.size(0));
    thrust::device_vector<T> output_counts(inputs.size(0));
    thrust::device_vector<T> total_prefix(inputs.size(0));
    thrust::device_vector<T> input_prefix(inputs.size(0));
    const T *ptr;
    size_t bs;
    ptr = count.data_ptr<T>();
    bs = inputs.size(0);
    thrust::copy(ptr, ptr + bs, input_prefix.begin());
    thrust::exclusive_scan(policy, input_prefix.begin(), input_prefix.end(),
                           input_prefix.begin());
    ptr = outputs.data_ptr<T>();
    bs = outputs.size(0);
    thrust::copy(ptr, ptr + bs, total_outputs.begin());
    const size_t m = inputs.size(0);
    using it = thrust::counting_iterator<T>;
    thrust::for_each(policy, it(0), it(m),
                     [input = thrust::raw_pointer_cast(total_inputs.data()),
                      count = thrust::raw_pointer_cast(output_counts.data()),
                      pre = thrust::raw_pointer_cast(input_prefix.data()),
                      prefix = thrust::raw_pointer_cast(total_prefix.data()),
                      in = inputs.data_ptr<T>(), cnt = count.data_ptr<T>(),
                      order = orders.data_ptr<T>()] __device__(T i) {
                         input[order[i]] = in[i];
                         count[order[i]] = cnt[i];
                         prefix[order[i]] = pre[i];
                     });

    thrust::device_vector<T> subset;
    TorchQuiver::reindex_kernel(stream, total_inputs, total_outputs, subset);

    int tot = total_outputs.size();
    torch::Tensor out_vertices = torch::empty(subset.size(), inputs.options());
    torch::Tensor row_idx = torch::empty(tot, inputs.options());
    torch::Tensor col_idx = torch::empty(tot, inputs.options());
    {
        thrust::device_vector<T> seq(output_counts.size());
        thrust::sequence(policy, seq.begin(), seq.end());

        thrust::for_each(
            policy, it(0), it(m),
            [prefix = thrust::raw_pointer_cast(total_prefix.data()),
             count = thrust::raw_pointer_cast(output_counts.data()),
             in = thrust::raw_pointer_cast(seq.data()),
             out = thrust::raw_pointer_cast(
                 row_idx.data_ptr<T>())] __device__(T i) {
                for (int j = 0; j < count[i]; j++) {
                    out[prefix[i] + j] = in[i];
                }
            });
        thrust::copy(subset.begin(), subset.end(), out_vertices.data_ptr<T>());
        thrust::copy(total_outputs.begin(), total_outputs.end(),
                     col_idx.data_ptr<T>());
    }
    return std::make_tuple(out_vertices, row_idx, col_idx);
}

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

/** add sample subgraph here **/
__global__ void
uniform_saintgraph_kernel(const int64_t *idx, const int64_t *rowptr,
                          const int64_t *row, const int64_t *col,
                          const int64_t *assoc,
                          thrust::tuple<int64_t, int64_t, int64_t> *edge_ptr,
                          int64_t *pre_sum, size_t num_of_sampled_node)
{
    const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx < num_of_sampled_node) {
        const int64_t output_idx = pre_sum[thread_idx];
        int64_t w, w_new, row_start, row_end;
        int64_t cur = idx[thread_idx];
        row_start = rowptr[cur], row_end = rowptr[cur + 1];
        int count = 0;
        for (int64_t j = row_start; j < row_end; j++) {
            w = col[j];
            w_new = assoc[w];
            edge_ptr[output_idx + count] =
                thrust::make_tuple<int64_t, int64_t, int64_t>(thread_idx, w_new,
                                                              j);
            count++;
        }
    }
}

struct is_sampled {
    __host__ __device__ bool
    operator()(const thrust::tuple<int64_t, int64_t, int64_t> &t)
    {
        return (thrust::get<1>(t)) == (int64_t)-1;
    }
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
saint_subgraph(const torch::Tensor &idx, const torch::Tensor &rowptr,
               const torch::Tensor &row, const torch::Tensor &col,
               const torch::Tensor &deg)
{
    CHECK_CUDA(idx);
    CHECK_CUDA(rowptr);
    CHECK_CUDA(col);
    CHECK_CUDA(deg);
    CHECK_INPUT(idx.dim() == 1);
    CHECK_INPUT(rowptr.dim() == 1);
    CHECK_INPUT(col.dim() == 1);
    CHECK_INPUT(deg.dim() == 1);
    const size_t num_of_edges = row.size(0);
    const size_t num_of_sampled_node = idx.size(0);
    const size_t num_of_nodes = rowptr.size(0);
    cudaStream_t stream = 0;
    const auto policy = thrust::cuda::par.on(stream);

    // input begin is what -> device ptr
    // input end is what -> device ptr
    // cast the idx to device ptr
    thrust::device_ptr<int64_t> idx_ptr_t =
        thrust::device_pointer_cast(idx.data_ptr<int64_t>());
    thrust::device_ptr<int64_t> output_counts =
        thrust::device_pointer_cast(deg.data_ptr<int64_t>());
    // presum array
    thrust::device_vector<int64_t> output_ptr;
    output_ptr.resize(num_of_sampled_node);

    thrust::exclusive_scan(policy, output_counts,
                           output_counts + num_of_sampled_node,
                           output_ptr.begin());

    int64_t num_sampled_edge = output_ptr[num_of_sampled_node - 1] +
                               output_counts[num_of_sampled_node - 1];
    auto assoc = torch::full({rowptr.size(0) - 1}, -1, idx.options());
    assoc.index_copy_(0, idx, torch::arange(idx.size(0), idx.options()));
    thrust::device_vector<thrust::tuple<int64_t, int64_t, int64_t>> edges(
        num_sampled_edge);

    // cast raw pointer*
    thrust::tuple<int64_t, int64_t, int64_t> *edge_ptr =
        thrust::raw_pointer_cast(&edges[0]);
    int64_t *presum_ptr = thrust::raw_pointer_cast(output_ptr.data());

    int threads = 1024;
    uniform_saintgraph_kernel<<<(idx.numel() + threads - 1) / threads, threads,
                                0, stream>>>(
        idx.data_ptr<int64_t>(), rowptr.data_ptr<int64_t>(),
        row.data_ptr<int64_t>(), col.data_ptr<int64_t>(),
        assoc.data_ptr<int64_t>(), edge_ptr, presum_ptr, idx.numel());

    // remove if not sampled
    auto new_end = thrust::remove_if(edges.begin(), edges.end(), is_sampled());
    edges.erase(new_end, edges.end());

    // copy
    torch::Tensor ret_row = torch::empty(edges.size(), idx.options());
    torch::Tensor ret_col = torch::empty(edges.size(), idx.options());
    torch::Tensor ret_indice = torch::empty(edges.size(), idx.options());

    thrust::transform(policy, edges.begin(), edges.end(),
                      ret_row.data_ptr<int64_t>(), thrust_get<0>());
    thrust::transform(policy, edges.begin(), edges.end(),
                      ret_col.data_ptr<int64_t>(), thrust_get<1>());
    thrust::transform(policy, edges.begin(), edges.end(),
                      ret_indice.data_ptr<int64_t>(), thrust_get<2>());

    return std::make_tuple(ret_row, ret_col, ret_indice);
}
}  // namespace quiver

void register_cuda_quiver(pybind11::module &m)
{
    m.def("saint_subgraph", &quiver::saint_subgraph);
    m.def("reindex_all", &quiver::reindex_all);
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
