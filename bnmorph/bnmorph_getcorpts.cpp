#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> bnmorph_cuda(
    torch::Tensor binMapsrc,
    torch::Tensor binMapdst,
    torch::Tensor xx,
    torch::Tensor yy,
    torch::Tensor sxx,
    torch::Tensor syy,
    torch::Tensor cxx,
    torch::Tensor cyy,
    float pixel_distance_weight,
    float alpha_distance_weight,
    float pixel_mulline_distance_weight,
    float alpha_padding
    );

// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> bnmorph(
    torch::Tensor binMapsrc,
    torch::Tensor binMapdst,
    torch::Tensor xx,
    torch::Tensor yy,
    torch::Tensor sxx,
    torch::Tensor syy,
    torch::Tensor cxx,
    torch::Tensor cyy,
    float pixel_distance_weight,
    float alpha_distance_weight,
    float pixel_mulline_distance_weight,
    float alpha_padding
    ) {
    CHECK_INPUT(binMapsrc)
    CHECK_INPUT(binMapdst)
    CHECK_INPUT(xx)
    CHECK_INPUT(yy)
    CHECK_INPUT(sxx)
    CHECK_INPUT(syy)
    CHECK_INPUT(cxx)
    CHECK_INPUT(cyy)
    std::vector<torch::Tensor> results_bindings = bnmorph_cuda(binMapsrc, binMapdst, xx, yy, sxx, syy, cxx, cyy, pixel_distance_weight, alpha_distance_weight, pixel_mulline_distance_weight, alpha_padding);
    return results_bindings;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bnmorph", &bnmorph, "a more general Beier–Neely morph");
}
