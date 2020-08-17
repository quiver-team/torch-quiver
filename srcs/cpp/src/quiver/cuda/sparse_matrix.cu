#include <algorithm>
#include <numeric>
#include <string>

#include <thrust/device_vector.h>

#include <torch/extension.h>

#include <quiver/common.hpp>
#include <quiver/quiver.cu.hpp>
#include <quiver/timer.hpp>
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
        thrust::device_vector<cuda_pair<int, int>> edges(n);
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

using torch_quiver_t = quiver<int64_t, CUDA>;
class TorchQuiver : public torch_quiver_t
{
    using torch_quiver_t::torch_quiver_t;

  public:
    using T = int64_t;

    torch::Tensor sample_adj(const torch::Tensor &vertices, int k)
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
        const size_t tot1 =
            compact(output_counts.size(), (size_t)k, output_counts.data(),
                    outputs.data(), out_vertices.data_ptr<T>());
        check_eq<T>(tot, tot1);
        return out_vertices;
    }
};

TorchQuiver new_quiver_from_edge_index(size_t n,
                                       const torch::Tensor &edge_index)
{
    timer _(__func__);
    using T = typename TorchQuiver::T;
    check(edge_index.is_contiguous());
    check_eq<int64_t>(edge_index.dim(), 2);
    check_eq<int64_t>(edge_index.size(0), 2);
    const size_t m = edge_index.size(1);
    const T *p = edge_index.data_ptr<T>();
    using vec = std::vector<std::pair<T, T>>;
    vec ei(m);
    {
        timer _("zip edge_index");
        zip(p, p + m, p + m, &ei[0].first);
    }
    return TorchQuiver((T)n, std::move(ei));
}
}  // namespace quiver

void register_sparse_matrix_cuda(pybind11::module &m)
{
    py::class_<quiver::SparseMatrixCuda>(m, "SparseMatrixCuda");
    m.def("new_sparse_matrix_cuda", &quiver::new_sparse_matrix_cuda);

    py::class_<quiver::TorchQuiver>(m, "Quiver")
        .def("sample_adj", &quiver::TorchQuiver::sample_adj);
    m.def("new_quiver_from_edge_index", &quiver::new_quiver_from_edge_index);
}
