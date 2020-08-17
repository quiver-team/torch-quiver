#include <algorithm>
#include <string>

#include <torch/extension.h>

#include <quiver/common.hpp>
#include <quiver/trace.hpp>

namespace quiver
{
class SparseMatrix
{
  public:
    SparseMatrix(const torch::Tensor &coo) {}
    ~SparseMatrix() {}
};

template <typename T>
SparseMatrix new_sparse_matrix_(const T *row_idx, const T *col_idx, size_t n,
                                const torch::Tensor &_)
{
    TRACE(__func__);
    {
        TRACE("min/max");
        std::cerr << "max(row_idx): " << *std::max_element(row_idx, row_idx + n)
                  << std::endl;
        std::cerr << "min(row_idx): " << *std::min_element(row_idx, row_idx + n)
                  << std::endl;
        std::cerr << "max(col_idx): " << *std::max_element(col_idx, col_idx + n)
                  << std::endl;
        std::cerr << "min(col_idx): " << *std::min_element(col_idx, col_idx + n)
                  << std::endl;
        std::cerr << "n: " << n << std::endl;
    }
    {
        TRACE("pair/sort");
        std::vector<std::pair<T, T>> edges(n);
        {
            TRACE("pair");
            for (size_t i = 0; i < n; ++i) {
                edges[i].first = row_idx[i];
                edges[i].second = col_idx[i];
            }
        }
        {
            TRACE("sort");
            std::sort(edges.begin(), edges.end());
        }
    }
    {
        TRACE("vertices");
        std::set<T> vertices;
        vertices.insert(row_idx, row_idx + n);
        vertices.insert(col_idx, col_idx + n);
        std::cerr << "support: " << vertices.size() << std::endl;
    }
    SparseMatrix spm(_);
    return spm;
}

SparseMatrix new_sparse_matrix(const torch::Tensor &coo)
{
    check(coo.is_contiguous());
    check_eq<long>(coo.dim(), 2);
    check_eq<int64_t>(coo.size(0), 2);
    const auto n = coo.size(1);
    const auto dtype = coo.dtype();
    if (dtype.Match<int>()) {
        const int *p = coo.data_ptr<int>();
        return new_sparse_matrix_<int>(p, p + n, n, coo);
    }
    if (dtype.Match<long>()) {
        const long *p = coo.data_ptr<long>();
        return new_sparse_matrix_<long>(p, p + n, n, coo);
    }
    throw std::runtime_error(std::string("unsupported type: ") +
                             static_cast<std::string>(dtype.name()));
}
}  // namespace quiver

void register_sparse_matrix(pybind11::module &m)
{
    // TypeError: Unable to convert function return value to a Python type! The
    // signature was
    //     (arg0: at::Tensor) -> quiver::SparseMatrix
    py::class_<quiver::SparseMatrix>(m, "SparseMatrix");
    m.def("new_sparse_matrix", &quiver::new_sparse_matrix);
}
