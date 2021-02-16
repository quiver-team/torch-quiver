#include <unordered_set>

#include <quiver/quiver.cpu.hpp>
#include <quiver/quiver.hpp>

namespace quiver
{
Quiver::Quiver() {}

std::unique_ptr<Quiver>
Quiver::from_edge_index_cpu(int n, std::vector<std::pair<int, int>> edge_index)
{
    using T = int64_t;
    using Q = quiver<T, CPU>;
    using vec = std::vector<std::pair<T, T>>;
    vec ei(edge_index.size());
    std::copy(edge_index.begin(), edge_index.end(), ei.begin());
    edge_index.clear();
    return std::make_unique<Q>(n, std::move(ei));
}

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
        return quiver_.sample(vertices, k);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    reindex_group(torch::Tensor prev_inputs, torch::Tensor inputs,
                  torch::Tensor counts, torch::Tensor outputs)
    {
        std::unordered_map<T, T> out_map;
        size_t in_size = inputs.size(0);
        size_t out_size = outputs.size(0);

        T *in_vec = inputs.data_ptr<T>();
        T *prev_in_vec = prev_inputs.data_ptr<T>();
        T *cnt_vec = counts.data_ptr<T>();
        T *out_vec = outputs.data_ptr<T>();

        std::vector<T> frontier;
        T n_id = 0;
        for (int i = 0; i < in_size; i++) {
            T input = in_vec[i];
            out_map[input] = n_id++;
            frontier.push_back(input);
        }
        for (int i = 0; i < out_size; i++) {
            T output = out_vec[i];
            if (out_map.find(output) == out_map.end()) {
                out_map[output] = n_id++;
                frontier.push_back(output);
            }
        }
        std::vector<T> row_idx(out_size);
        std::vector<T> col_idx(out_size);
        size_t cnt = 0;
        for (int i = 0; i < in_size; i++) {
            T in = prev_in_vec[i];
            for (int j = 0; j < cnt_vec[i]; j++) {
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
    for (int i = 0; i < e; i++) {
        ei[i].first = *ptr;
        ptr++;
    }
    for (int i = 0; i < e; i++) {
        ei[i].second = *ptr;
        ptr++;
    }
    Q quiver(n, std::move(ei));
    return CPUQuiver(std::move(quiver));
}
}  // namespace quiver

void register_cpu_quiver(pybind11::module &m)
{
    m.def("cpu_quiver_from_edge_index", &quiver::cpu_quiver_from_edge_index);
    py::class_<quiver::CPUQuiver>(m, "CPUQuiver")
        .def("sample_neighbor", &quiver::CPUQuiver::sample_neighbor)
        .def("reindex_group", &quiver::CPUQuiver::reindex_group);
}
