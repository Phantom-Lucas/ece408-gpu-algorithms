#include <cstdio>
#include <cstdlib>

#include "helper.hpp"

#define TILE_SIZE 30

__global__ void kernel(int *A0, int *Anext, int nx, int ny, int nz) {

  if(nz<3)return;
  __shared__ int smem[TILE_SIZE+2][TILE_SIZE+2];
  int x = blockIdx.x * TILE_SIZE + threadIdx.x; 
  int y = blockIdx.y * TILE_SIZE + threadIdx.y;
  int tx= threadIdx.x;
  int ty= threadIdx.y;
  int prev = 0;
  int curr = 0;
  int next = 0;
  if (x < nx && y < ny) {
    prev = A0[y * nx + x];
    curr = A0[nx * ny + y * nx + x];
    next = A0[2 * nx * ny + y * nx + x];
  }
  for(int z=1; z<nz-1; z++) {
    if (x < nx && y < ny) {
      smem[ty][tx] = curr;
    } else {
      smem[ty][tx] = 0;
    }
    __syncthreads();
    int sum=0;
    if (tx>0 && tx<TILE_SIZE+1 && ty>0 && ty<TILE_SIZE+1 && x<nx-1 && y<ny-1) {
      sum = smem[ty-1][tx] + smem[ty+1][tx] + smem[ty][tx-1] + smem[ty][tx+1] + prev + next- 6 * smem[ty][tx];
      Anext[z * (ny * nx) + y * nx + x] = sum;
    }
    __syncthreads();
    prev = curr;
    curr = next;
    if (z < nz - 2 && x < nx && y < ny) {
      next = A0[(z+2) * nx * ny + y * nx + x];
    }
  }
}

void launchStencil(int* A0, int* Anext, int nx, int ny, int nz) {

  dim3 block(TILE_SIZE+2, TILE_SIZE+2);
  int gridSizeX = (nx -2 + TILE_SIZE - 1) / TILE_SIZE;
  int gridSizeY = (ny -2 + TILE_SIZE - 1) / TILE_SIZE;
  dim3 grid(gridSizeX, gridSizeY);  
  kernel<<<grid, block>>>(A0, Anext, nx, ny, nz);

}


static int eval(const int nx, const int ny, const int nz) {

  // Generate model
  const auto conf_info = std::string("stencil[") + std::to_string(nx) + "," + 
                                                   std::to_string(ny) + "," + 
                                                   std::to_string(nz) + "]";
  INFO("Running "  << conf_info);

  // generate input data
  timer_start("Generating test data");
  std::vector<int> hostA0(nx * ny * nz);
  generate_data(hostA0.data(), nx, ny, nz);
  std::vector<int> hostAnext(nx * ny * nz);

  timer_start("Allocating GPU memory.");
  int *deviceA0 = nullptr, *deviceAnext = nullptr;
  CUDA_RUNTIME(cudaMalloc((void **)&deviceA0, nx * ny * nz * sizeof(int)));
  CUDA_RUNTIME(cudaMalloc((void **)&deviceAnext, nx * ny * nz * sizeof(int)));
  timer_stop();

  timer_start("Copying inputs to the GPU.");
  CUDA_RUNTIME(cudaMemcpy(deviceA0, hostA0.data(), nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU convlayer");
  launchStencil(deviceA0, deviceAnext, nx, ny, nz);
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  timer_start("Copying output to the CPU");
  CUDA_RUNTIME(cudaMemcpy(hostAnext.data(), deviceAnext, nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  // verify with provided implementation
  timer_start("Verifying results");
  verify(hostAnext.data(), hostA0.data(), nx, ny, nz);
  timer_stop();

  CUDA_RUNTIME(cudaFree(deviceA0));
  CUDA_RUNTIME(cudaFree(deviceAnext));

  return 0;
}



TEST_CASE("Stencil", "[stencil]") {

  SECTION("[dims:32,32,32]") {
    eval(32,32,32);
  }
  SECTION("[dims:30,30,30]") {
    eval(30,30,30);
  }
  SECTION("[dims:29,29,29]") {
    eval(29,29,29);
  }
  SECTION("[dims:31,31,31]") {
    eval(31,31,31);
  }
  SECTION("[dims:29,29,2]") {
    eval(29,29,2);
  }
  SECTION("[dims:512,512,64]") {
    eval(512,512,64);
  }

}
