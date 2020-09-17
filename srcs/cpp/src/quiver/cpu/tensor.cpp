#include <climits>
#include <iostream>
#include <numeric>
#include <string>

// #ifdef HAVE_CXXABI
#include <cxxabi.h>
// #endif

#include <torch/extension.h>

namespace quiver
{
template <typename T>
std::string demangled_type_info_name()
{
#ifdef HAVE_CXXABI
    int status = 0;
    return abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
#else
    return typeid(T).name();
#endif
}

class TensorShape
{
    std::vector<int64_t> dims_;

  public:
    void AddDim(int d);

    std::string str() const;

    const std::vector<int64_t> &Dims() const;

    int64_t size() const;
};

void TensorShape::AddDim(int d) { dims_.push_back(d); }

const std::vector<int64_t> &TensorShape::Dims() const { return dims_; }

std::string TensorShape::str() const
{
    std::string s = "(";
    for (size_t i = 0; i < dims_.size(); ++i) {
        if (i > 0) { s += ", "; }
        s += std::to_string(dims_[i]);
    }
    s += ")";
    return s;
}

int64_t TensorShape::size() const
{
    return std::accumulate(dims_.begin(), dims_.end(), static_cast<int64_t>(1),
                           std::multiplies<int64_t>());
}

TensorShape get_tensor_shape(const torch::Tensor &x)
{
    TensorShape shape;
    for (int idx = 0; idx < x.dim(); ++idx) { shape.AddDim(x.size(idx)); }
    return shape;
}

template <typename T>
void matched()
{
    std::cerr << "matched: " << demangled_type_info_name<T>()
              << ", sizeof(T) = " << sizeof(T) << std::endl;
};

void show_tensor_info(const torch::Tensor &t)
{
    const auto shape = get_tensor_shape(t);
    const auto dtype = t.dtype();
    std::cerr << "shape: " << shape.str()             //
              << ", dtype: " << dtype                 //
              << ", dtypeid: " << dtype.id()          //
              << ", size: " << t.numel()              //
              << ", elem_size: " << t.element_size()  //
              << std::endl;

    std::cerr << "demangled_type_info_name: "
              << demangled_type_info_name<decltype(dtype)>() << std::endl;

    if (dtype.Match<int>()) { matched<int>(); }
    if (dtype.Match<long>()) { matched<long>(); }
    if (dtype.Match<float>()) { matched<float>(); }
    if (dtype.Match<double>()) { matched<double>(); }

    const auto dev = t.device();
    std::cerr << "device: " << dev << std::endl;
}
}  // namespace quiver
