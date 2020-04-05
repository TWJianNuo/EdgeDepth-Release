#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>


namespace {

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  return fmaxf(0.0, z) + fminf(0.0, alpha * (exp(z) - 1.0));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1 / (1 + std::exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t alpha_d_weight_function(scalar_t x, scalar_t sr) {
    // sr for smooth range
    scalar_t sigmoid_sr = 12.0 / sr;
    scalar_t bias_pos = sr / 2.0 + 1.0;
    scalar_t y;
    if ((x < 1) && (x > 0)){
        y = 1;
    }
    else if(x <= 0){
        y = 0;
    }
    else{
        y = sigmoid(-sigmoid_sr * (x - bias_pos));

    }
    return y;
}

template <typename scalar_t>
__global__ void bnmorph_cuda_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> binMapsrc,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> binMapdst,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> orgpts_x,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> orgpts_y,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> correspts_x,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> correspts_y,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> xx,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> yy,
    const int searchlen,
    const int ele_channel_num,
    const int height,
    const int width) {

    const int linear_index_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int look_x;
    int look_y;
    if (linear_index_pos < ele_channel_num){
        int dimy = linear_index_pos / width;
        int dimx = linear_index_pos - dimy * width;
        if (std::abs(binMapsrc[blockIdx.y][0][dimy][dimx] - 1.0) < 1e-5){
            for(int i = 0; i < searchlen; i++){
                look_x = dimx + int(xx[i]);
                look_y = dimy + int(yy[i]);
                if((look_x >= 0) && (look_y >= 0) && (look_x < width) && (look_y < height)){
                    if(std::abs(binMapdst[blockIdx.y][0][look_y][look_x] - 1.0) < 1e-5){
                        orgpts_x[blockIdx.y][0][dimy][dimx] = dimx;
                        orgpts_y[blockIdx.y][0][dimy][dimx] = dimy;
                        correspts_x[blockIdx.y][0][dimy][dimx] = look_x;
                        correspts_y[blockIdx.y][0][dimy][dimx] = look_y;
                        break;
                    }
                }
            }
        }
    }

}


template <typename scalar_t>
__global__ void bnmorph_suppress_for_sparsity(
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> orgpts_x,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> orgpts_y,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> correspts_x,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> correspts_y,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> sxx,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> syy,
    const int ele_channel_num,
    const int height,
    const int width,
    const int sparse_height,
    const int sparse_width,
    const int sparse_rec_len,
    const int sparse_check_len,
    const int sparse_check_startpos
    ) {

    const int sparse_grid_linear_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if(sparse_grid_linear_ind < sparse_height * sparse_width){
        int sparse_grid_y = sparse_grid_linear_ind / sparse_width;
        int sparse_grid_x = sparse_grid_linear_ind - sparse_grid_y * sparse_width;

        int image_grid_start_x = sparse_grid_x * sparse_rec_len;
        int image_grid_start_y = sparse_grid_y * sparse_rec_len;
        int bias = (sparse_rec_len - 1) / 2;

        int image_grid_lookupx;
        int image_grid_lookupy;

        int modded_index;
        bool isFirst = true;

        for(int i = 0; i < sparse_check_len; i++){
            modded_index = (sparse_check_startpos + i) % sparse_check_len;
            image_grid_lookupx = image_grid_start_x + sxx[modded_index] + bias;
            image_grid_lookupy = image_grid_start_y + syy[modded_index] + bias;

            if((image_grid_lookupx >= 0) && (image_grid_lookupy >= 0) && (image_grid_lookupx < width) && (image_grid_lookupy < height)){
                if(orgpts_x[blockIdx.y][0][image_grid_lookupy][image_grid_lookupx] > -1e-5){
                    if(isFirst){
                        isFirst = false;
                    }
                    else{
                        orgpts_x[blockIdx.y][0][image_grid_lookupy][image_grid_lookupx] = -1.0;
                        orgpts_y[blockIdx.y][0][image_grid_lookupy][image_grid_lookupx] = -1.0;
                        correspts_x[blockIdx.y][0][image_grid_lookupy][image_grid_lookupx] = -1.0;
                        correspts_y[blockIdx.y][0][image_grid_lookupy][image_grid_lookupx] = -1.0;
                    }
                }
            }
        }
    }
    }


template <typename scalar_t>
__global__ void bnmorph_morph_coord(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> orgpts_x,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> orgpts_y,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> correspts_x,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> correspts_y,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cxx,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cyy,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> morphedx,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> morphedy,
    const int height,
    const int width,
    const int morph_search_length,
    float pixel_distance_weight,
    float alpha_distance_weight,
    float pixel_mulline_distance_weight,
    float alpha_padding
    ) {

    const int linear_index_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int batchind = blockIdx.y;
    float storedDat[200][3] = {};
    const int maxlineNum = 200;
    const float eratio = 1.0;
    const float fixratio = -(1.0 / (1.0 + eratio));
    if (linear_index_pos < width * height){
        int dimy = linear_index_pos / width;
        int dimx = linear_index_pos - dimy * width;
        int lookupx;
        int lookupy;
        int count = 0;

        float orgx;
        float orgy;
        float dstx;
        float dsty;
        float orgrootx;
        float orgrooty;

        float dorg;
        float ddst;
        float dline;

        float alpha;
        float over_dline;

        float morphed_x;
        float morphed_y;

        float alpha_d_weight;
        float pixel_d_mulline_weight;

        float aqx;
        float aqy;
        float aqscale;


        for(int i = 0; i < morph_search_length; i++){
            lookupx = cxx[i] + dimx;
            lookupy = cyy[i] + dimy;
            if((lookupx > 0) && (lookupy > 0) && (lookupx < width) && (lookupy < height) && (count < maxlineNum)){
                if(orgpts_x[batchind][0][lookupy][lookupx] > -1e-3){
                    // Start computation
                    dimx = (float)dimx;
                    dimy = (float)dimy;

                    orgx = orgpts_x[batchind][0][lookupy][lookupx];
                    orgy = orgpts_y[batchind][0][lookupy][lookupx];
                    dstx = correspts_x[batchind][0][lookupy][lookupx];
                    dsty = correspts_y[batchind][0][lookupy][lookupx];

                    if((orgx - dstx)*(orgx - dstx) + (orgy - dsty)*(orgy - dsty) > 0){

                        dorg = std::sqrt((orgx - dimx) * (orgx - dimx) + (orgy - dimy) * (orgy - dimy));
                        ddst = std::sqrt((dstx - dimx) * (dstx - dimx) + (dsty - dimy) * (dsty - dimy));
                        dline = std::abs( (dstx - orgx) * (orgy - dimy) - (orgx - dimx) * (dsty - orgy) ) / (std::sqrt( (orgx - dstx)*(orgx - dstx) + (orgy - dsty)*(orgy - dsty) ) + 1e-5);


                        orgrootx = dstx + (1.0 + eratio) * (orgx - dstx);
                        orgrooty = dsty + (1.0 + eratio) * (orgy - dsty);


                        alpha = ((dimx - orgrootx) * (dstx - orgrootx) + (dimy - orgrooty) * (dsty - orgrooty)) / ((orgrootx - dstx)*(orgrootx - dstx) + (orgrooty - dsty)*(orgrooty - dsty) + 1e-5);

                        if((alpha < 0) || (alpha > 1)){
                            if(ddst < dorg){
                                over_dline = ddst;
                            }
                            else{
                                over_dline = dorg;
                            }
                        }
                        else{
                            over_dline = dline;
                        }

                        alpha_d_weight = alpha_d_weight_function(alpha, alpha_distance_weight);

                        pixel_d_mulline_weight = std::pow(1 / (over_dline + alpha_padding), pixel_mulline_distance_weight);

                        aqscale = ((dimx - orgrootx) * (dstx - orgx) + (dimy - orgrooty) * (dsty - orgy)) / ((dstx - orgx)*(dstx - orgx) + (dsty - orgy)*(dsty - orgy) + 1e-5);
                        aqx = (dstx - orgx) * aqscale;
                        aqy = (dsty - orgy) * aqscale;

                        morphed_x = alpha_d_weight * fixratio * aqx;
                        morphed_y = alpha_d_weight * fixratio * aqy;

                        storedDat[count][0] = pixel_d_mulline_weight;
                        storedDat[count][1] = morphed_x;
                        storedDat[count][2] = morphed_y;
                        count = count + 1;
                    }

                }
            }
        }

        dimx = (int)dimx;
        dimy = (int)dimy;

        if(count == 0){
            morphedx[batchind][0][dimy][dimx] = dimx;
            morphedy[batchind][0][dimy][dimx] = dimy;
        }
        else{
            float averaged_x = 0;
            float averaged_y = 0;
            float totweight = 1e-5;
            for(int t = 0; t < count; t++){
                totweight = totweight + storedDat[t][0];
            }
            for(int t = 0; t < count; t++){
                averaged_x = averaged_x + storedDat[t][1] * storedDat[t][0] / totweight;
                averaged_y = averaged_y + storedDat[t][2] * storedDat[t][0] / totweight;
            }
            morphedx[batchind][0][dimy][dimx] = averaged_x + (float)dimx;
            morphedy[batchind][0][dimy][dimx] = averaged_y + (float)dimy;
        }

    }



    }

} // namespace




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
    ) {
  const int batch_size = binMapsrc.size(0);
  const int height = binMapsrc.size(2);
  const int width = binMapsrc.size(3);
  const int searchlen = xx.size(0);
  const int ele_channel_num = height * width;
  const int threads = 1024;
  const dim3 blocks((ele_channel_num + threads - 1) / threads, binMapsrc.size(0));
  const int sparse_check_len = sxx.size(0);
  const int sparse_check_startpos = (std::rand()) % sparse_check_len;
  const int sparse_rec_len = int(std::sqrt(sparse_check_len));

  auto orgpts_x = (torch::ones_like(binMapsrc)) * -1;
  auto orgpts_y = (torch::ones_like(binMapsrc)) * -1;
  auto correspts_x = (torch::ones_like(binMapsrc)) * -1;
  auto correspts_y = (torch::ones_like(binMapsrc)) * -1;
  auto sparse_indicator = torch::ones_like(binMapsrc);

  AT_DISPATCH_FLOATING_TYPES(binMapsrc.type(), "find corresponding pts cuda", ([&] {
    bnmorph_cuda_kernel<scalar_t><<<blocks, threads>>>(
        binMapsrc.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        binMapdst.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        orgpts_x.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        orgpts_y.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        correspts_x.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        correspts_y.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        xx.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        yy.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        searchlen,
        ele_channel_num,
        height,
        width
        );
  }));

  const int sparse_width = std::ceil((float)width / sparse_rec_len);
  const int sparse_height = std::ceil((float)height / sparse_rec_len);
  const int threads_sparse = 1024;
  const dim3 blocks_sparse((sparse_height * sparse_width + threads - 1) / threads, batch_size);


  AT_DISPATCH_FLOATING_TYPES(binMapsrc.type(), "suppress sparsity cuda", ([&] {
    bnmorph_suppress_for_sparsity<scalar_t><<<blocks_sparse, threads_sparse>>>(
        orgpts_x.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        orgpts_y.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        correspts_x.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        correspts_y.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        sxx.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        syy.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        ele_channel_num,
        height,
        width,
        sparse_height,
        sparse_width,
        sparse_rec_len,
        sparse_check_len,
        sparse_check_startpos
        );
  }));

  auto orgpts_x_cpu = orgpts_x.to(torch::kCPU);
  auto orgpts_y_cpu = orgpts_y.to(torch::kCPU);
  auto correspts_x_cpu = correspts_x.to(torch::kCPU);
  auto correspts_y_cpu = correspts_y.to(torch::kCPU);

  auto orgpts_x_cpu_iter = orgpts_x_cpu.accessor<float, 4>();
  auto orgpts_y_cpu_iter = orgpts_y_cpu.accessor<float, 4>();
  auto correspts_x_cpu_iter = correspts_x_cpu.accessor<float, 4>();
  auto correspts_y_cpu_iter = correspts_y_cpu.accessor<float, 4>();

  auto morphedx = torch::zeros_like(binMapsrc);
  auto morphedy = torch::zeros_like(binMapsrc);
  const int morph_search_length = cxx.size(0);
  const int threads_morph = 1024;
  const dim3 blocks_morph((ele_channel_num + threads - 1) / threads, binMapsrc.size(0));
  AT_DISPATCH_FLOATING_TYPES(binMapsrc.type(), "coordinate morph cuda", ([&] {
    bnmorph_morph_coord<scalar_t><<<blocks_morph, threads_morph>>>(
        orgpts_x.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        orgpts_y.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        correspts_x.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        correspts_y.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        cxx.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        cyy.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        morphedx.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        morphedy.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        height,
        width,
        morph_search_length,
        pixel_distance_weight,
        alpha_distance_weight,
        pixel_mulline_distance_weight,
        alpha_padding
        );
  }));

  return {orgpts_x, orgpts_y, correspts_x, correspts_y, morphedx, morphedy};
}

