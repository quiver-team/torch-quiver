#include <unordered_map>

#include <quiver/quiver.cpu.hpp>
#include <quiver/quiver.hpp>

#include <torch/extension.h>
#include <quiver/common.hpp>

namespace quiver
{
class CPUQuiver
{
    using torch_quiver_t = quiver<int64_t, CPU>;
    torch_quiver_t quiver_;

  public:
    CPUQuiver(torch_quiver_t quiver) : quiver_(std::move(quiver)) {}

    using T = int64_t;

    std::tuple<torch::Tensor, torch::Tensor>
    sample_neighbor(const torch::Tensor &vertices, int k)
    {
        size_t bs = vertices.size(0);
        std::vector<T> inputs(bs);
        std::copy(vertices.data_ptr<T>(), vertices.data_ptr<T>() + bs,
                  inputs.begin());
        auto res = quiver_.sample_kernel(inputs, k);
        auto &outputs = std::get<0>(res);
        auto &output_counts = std::get<1>(res);
        size_t total = outputs.size();
        torch::Tensor out = torch::empty(total, vertices.options());
        torch::Tensor counts = torch::empty(bs, vertices.options());
        std::copy(outputs.begin(), outputs.end(), out.data_ptr<T>());
        std::copy(output_counts.begin(), output_counts.end(),
                  counts.data_ptr<T>());
        return std::make_tuple(out, counts);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    reindex_single(torch::Tensor inputs, torch::Tensor outputs, torch::Tensor counts)
    {
        std::unordered_map<T, T> out_map;
        size_t in_size = inputs.size(0);
        size_t out_size = outputs.size(0);

        T *in_vec = inputs.data_ptr<T>();
        //T *prev_in_vec = prev_inputs.data_ptr<T>();
        T *cnt_vec = counts.data_ptr<T>();
        T *out_vec = outputs.data_ptr<T>();

        std::vector<T> frontier;
        T n_id = 0;
        for (size_t i = 0; i < in_size; i++) {
            T input = in_vec[i];
            out_map[input] = n_id++;
            frontier.push_back(input);
        }
        for (size_t i = 0; i < out_size; i++) {
            T output = out_vec[i];
            if (out_map.find(output) == out_map.end()) {
                out_map[output] = n_id++;
                frontier.push_back(output);
            }
        }
        std::vector<T> row_idx(out_size);
        std::vector<T> col_idx(out_size);
        size_t cnt = 0;
        for (size_t i = 0; i < in_size; i++) {
            T in = in_vec[i];
            for (T j = 0; j < cnt_vec[i]; j++) {
                row_idx[cnt] = out_map[in];
                col_idx[cnt] = out_map[out_vec[cnt]];
                cnt++;
            }
        }
        torch::Tensor rows = torch::empty(cnt, inputs.options());
        torch::Tensor cols = torch::empty(cnt, inputs.options());
        torch::Tensor result = torch::empty(n_id, inputs.options());
        std::copy(row_idx.begin(), row_idx.end(), rows.data_ptr<T>());
        std::copy(col_idx.begin(), col_idx.end(), cols.data_ptr<T>());
        std::copy(frontier.begin(), frontier.end(), result.data_ptr<T>());
        return std::make_tuple(result, rows, cols);
    }
};

CPUQuiver cpu_quiver_from_edge_index(size_t n, torch::Tensor edge_index)
{
    using T = int64_t;
    using Q = quiver<T, CPU>;
    using vec = std::vector<std::pair<T, T>>;
    size_t e = edge_index.size(1);
    vec ei(e);
    T *ptr = edge_index.data_ptr<T>();
    for (size_t i = 0; i < e; i++) {
        ei[i].first = *ptr;
        ptr++;
    }
    for (size_t i = 0; i < e; i++) {
        ei[i].second = *ptr;
        ptr++;
    }
    Q quiver(n, std::move(ei));
    return CPUQuiver(std::move(quiver));
}
CPUQuiver cpu_quiver_from_csr_array(torch::Tensor &input_indptr,
                                    torch::Tensor &input_indices)
{
    using T = int64_t;
    using Q = quiver<T, CPU>;
    check_eq<int64_t>(input_indptr.dim(), 1);
    const size_t node_count = input_indptr.size(0) - 1;

    check_eq<int64_t>(input_indices.dim(), 1);
    const size_t edge_count = input_indices.size(0);

    Q quiver(node_count, edge_count, input_indptr.data_ptr<T>(), input_indices.data_ptr<T>());
    return CPUQuiver(std::move(quiver));
}

}  // namespace quiver

void register_cpu_quiver(pybind11::module &m)
{
    m.def("cpu_quiver_from_edge_index", &quiver::cpu_quiver_from_edge_index);
    m.def("cpu_quiver_from_csr_array", &quiver::cpu_quiver_from_csr_array);
    py::class_<quiver::CPUQuiver>(m, "CPUQuiver")
        .def("sample_neighbor", &quiver::CPUQuiver::sample_neighbor)
        .def("reindex_single", &quiver::CPUQuiver::reindex_single);
}