#include "helper.hpp"


constexpr int MAX_W_SIZE = 32 * 1 * 5 * 5;
__constant__ float W_const[MAX_W_SIZE];
// Sequential code for the forward path of the convolution layer
// You should not modify this code
static void conv_forward_valid(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y,
                               const shape &ydims) {
  std::fill(Y, Y + ydims.flattened_length(), 0);

  for (auto i : range(0, ydims.num)) {
    for (auto m : range(0, ydims.depth )) {   // for each output feature map
      for (auto h : range(0, ydims.height)) { // for each output element
        for (auto w : range(0, ydims.width )) {
          const auto yoffset = ((i * ydims.depth + m) * ydims.height + h) * ydims.width + w;
          for (auto c : range(0, xdims.depth )) {     // sum over all input feature maps
            for (auto p : range(0, wdims.height)) {   // filter height
              for (auto q : range(0, wdims.width )) { // filter width
                const auto xoffset = ((((i * xdims.depth) + c) * xdims.height) + (h + p)) * xdims.width + (w + q);
                const auto woffset = ((((m * wdims.depth) + c) * wdims.height) + p) * wdims.width + q;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}

// Baseline GPU kernel code for forward convolution.
// One thread per output index
// You should not modify this kernel as it is used for correctness comparison.
// Instead, define a new one below
__global__ void conv_forward_baseline_kernel(const float *X, const shape xdims, const float *W, const shape wdims, float *Y,
                                    const shape ydims) {

  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = gx; i < ydims.num * ydims.depth * ydims.height * ydims.width; i += blockDim.x * gridDim.x) {
    Y[i] = 0.f;
  }

  for (size_t i = gx; i < ydims.num; i += gridDim.x * blockDim.x) {
    for (auto m : range(0, ydims.depth )) { // for each output feature map
      for (auto h : range(0, ydims.height)) { // for each output element
        for (auto w : range(0, ydims.width )) {
          const size_t yoffset = ((i * ydims.depth + m) * ydims.height + h) * ydims.width + w;
          for (auto c : range(0, xdims.depth )) {     // sum over all input feature maps
            for (auto p : range(0, wdims.height)) {   // filter height
              for (auto q : range(0, wdims.width )) { // filter width
                const size_t xoffset = ((((i * xdims.depth) + c) * xdims.height) + (h + p)) * xdims.width + (w + q);
                const size_t woffset = ((((m * wdims.depth) + c) * wdims.height) + p) * wdims.width + q;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}

// Host code to configure baseline GPU kernel
static void convlayer_gpu_baseline(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y,
  const shape &ydims) {

  dim3 dimGrid(1);
  dim3 dimBlock(32);

  conv_forward_baseline_kernel<<<dimGrid, dimBlock>>>(X, xdims, W, wdims, Y, ydims);
  THROW_IF_ERROR(cudaGetLastError());

}

// Implement your optimized kernel here.
// Make any modifications you wish.
// Don't forget to modify the host code below, if needed!
__global__ void conv_forward_opt_kernel(const float *X, const shape xdims, const float *W, const shape wdims, float *Y,
  const shape ydims) {

    extern __shared__ float smem[];
    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int tid=ty*blockDim.x+tx;
    int num_threads=blockDim.x*blockDim.y;

    int tile_height=blockDim.y+wdims.height-1;
    int tile_width=blockDim.x+wdims.width-1;

    int tile_size=tile_height*tile_width;
    int filter_size=wdims.depth*wdims.height*wdims.width;

    float *Xs=smem;
    float *Ws=smem+tile_size;

    //
    int h=blockIdx.y*blockDim.y+ty;
    int w=blockIdx.x*blockDim.x+tx;

    int i=blockIdx.z/(ydims.depth);
    int m=blockIdx.z%(ydims.depth);

    int base_h=blockIdx.y*blockDim.y;
    int base_w=blockIdx.x*blockDim.x;

    //
    for(int idx=tid;idx<filter_size;idx+=num_threads){
        Ws[idx]=W[m*filter_size+idx];
    }
    __syncthreads();

    //
    float sum=0.0f;
    for(int c=0;c<xdims.depth;c++){
      for(int idx=tid;idx<tile_size;idx+=num_threads){
        int sy=idx/tile_width;
        int sx=idx%tile_width;

        int xh=base_h+sy;
        int xw=base_w+sx;

        if(xh<xdims.height && xw<xdims.width){
          int xoffset=((i*xdims.depth+c)*xdims.height+xh)*xdims.width+xw;
          Xs[idx] = X[xoffset];
        }else{
          Xs[idx]=0.0f;
        }
      }
      __syncthreads();

      if(h<ydims.height && w<ydims.width){
        for(int p=0;p<wdims.height;p++){
          for(int q=0;q<wdims.width;q++){
            int xs_offset = (ty + p) * tile_width + (tx + q);
            int ws_offset=((c*wdims.height+p)*wdims.width+q);
            sum+=Xs[xs_offset]*Ws[ws_offset];
          }
        }
      }
      __syncthreads();
    
    }
    if(h<ydims.height && w<ydims.width){
      int yoffset=((i*ydims.depth+m)*ydims.height+h)*ydims.width+w;
      Y[yoffset]=sum;
    }
}

// Host code to configure baseline GPU kernel
static void convlayer_gpu_opt(
    const float *X,
    const shape &xdims,
    const float *W,
    const shape &wdims,
    float *Y,
    const shape &ydims)
{
    constexpr int TILE = 16;

    // 每个 block 16×16 = 256 threads
    dim3 dimBlock(TILE, TILE);

    // x/y 划分输出 feature map
    // z 同时表示 image i 和 output channel m
    dim3 dimGrid(
        (ydims.width  + TILE - 1) / TILE,
        (ydims.height + TILE - 1) / TILE,
        ydims.num * ydims.depth
    );

    /*
     * 一个 16×16 output tile 需要：
     *
     * (16 + Kh - 1) × (16 + Kw - 1)
     *
     * 的 input tile
     */
    int input_tile_height =
        TILE + wdims.height - 1;

    int input_tile_width =
        TILE + wdims.width - 1;

    int input_tile_size =
        input_tile_height * input_tile_width;

    /*
     * 当前 block 只负责一个 output channel m，
     * 所以只需要：
     *
     * W[m][:][:][:]
     *
     * 大小 = C × Kh × Kw
     */
    int filter_size =
        wdims.depth *
        wdims.height *
        wdims.width;

    /*
     * shared memory:
     *
     * | input tile | filter |
     */
    size_t shared_size =
        (input_tile_size + filter_size) *
        sizeof(float);

    conv_forward_opt_kernel<<<
        dimGrid,
        dimBlock,
        shared_size
    >>>(
        X,
        xdims,
        W,
        wdims,
        Y,
        ydims
    );

    THROW_IF_ERROR(cudaGetLastError());
}


static int eval(const shape wDims, const shape xDims) {

  // Generate model
  const auto conf_info = std::string("conv[wDims:") + std::to_string(wDims.num) + "," +
                                                      std::to_string(wDims.depth) + "," +
                                                      std::to_string(wDims.height) + "," +
                                                      std::to_string(wDims.width) +
                                                      " xDims:" + std::to_string(xDims.num) + "," +
                                                      std::to_string(xDims.depth) + "," +
                                                      std::to_string(xDims.height) + "," +
                                                      std::to_string(xDims.width) + "]";
  INFO("Running "  << conf_info);

  // Generate convolution weights
  float *hostW = allocate<float>(wDims);
  generate_convfilters(hostW, wDims);

  // generate input feature map
  float *hostX = allocate<float>(xDims);
  generate_data(hostX, xDims);

  // generate output feature map for verification
  const shape ydims = {xDims.num, wDims.num, (xDims.height - wDims.height + 1),
      (xDims.width - wDims.width + 1)};
  INFO("Allocating output tensor [" << ydims.num << "," << ydims.depth << "," << ydims.height << "," << ydims.width << "]");
  float *hostY = allocate<float>(ydims);
  float *expected = allocate<float>(ydims);
  generate_data(hostY, ydims);


  const size_t wByteCount = wDims.flattened_length() * sizeof(float);
  const size_t xByteCount = xDims.flattened_length() * sizeof(float);
  const size_t yByteCount = ydims.flattened_length() * sizeof(float);

  float *deviceW = nullptr, *deviceX = nullptr, *deviceY = nullptr;
  timer_start("Allocating GPU memory.");
  THROW_IF_ERROR(cudaMalloc((void **)&deviceW, wByteCount));
  THROW_IF_ERROR(cudaMalloc((void **)&deviceX, xByteCount));
  THROW_IF_ERROR(cudaMalloc((void **)&deviceY, yByteCount));
  timer_stop();


  timer_start("Copying inputs to the GPU.");
  THROW_IF_ERROR(cudaMemcpy(deviceW, hostW, wByteCount, cudaMemcpyDefault));
  THROW_IF_ERROR(cudaMemcpy(deviceX, hostX, xByteCount, cudaMemcpyDefault));
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU convlayer");
  convlayer_gpu_opt(deviceX, xDims, deviceW, wDims, deviceY, ydims);
  THROW_IF_ERROR(cudaDeviceSynchronize());
  timer_stop();

  timer_start("Copying output to the CPU");
  THROW_IF_ERROR(cudaMemcpy(hostY, deviceY, yByteCount, cudaMemcpyDefault));
  timer_stop();

  // verify with provided implementation
  convlayer_gpu_baseline(deviceX, xDims, deviceW, wDims, deviceY, ydims);
  THROW_IF_ERROR(cudaDeviceSynchronize());
  THROW_IF_ERROR(cudaMemcpy(expected, deviceY, yByteCount, cudaMemcpyDefault));
  verify(expected, hostY, ydims);

  THROW_IF_ERROR(cudaFree(deviceW));
  THROW_IF_ERROR(cudaFree(deviceX));
  THROW_IF_ERROR(cudaFree(deviceY));
  free(hostW);
  free(hostX);
  free(hostY);
  free(expected);

  return 0;
}



TEST_CASE("Convlayer", "[convlayer]") {
  SECTION("[wDims:0,0,0,0 xDims:100,1,32,32]") {
    eval({0,0,0,0}, {100,1,32,32});
  }
  SECTION("[wDims:1,1,1,1 xDims:100,1,32,32]") {
    eval({1,1,1,1}, {100,1,32,32});
  }
  SECTION("[wDims:32,1,5,5 xDims:1000,1,28,28]") {
    eval({32,1,5,5}, {1000,1,28,28});
  }
  SECTION("[wDims:16,1,3,3 xDims:100,1,32,32]") {
    eval({16,1,3,3}, {100,1,32,32});
  }

}
