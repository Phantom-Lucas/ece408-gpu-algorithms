#include <cstdio>
#include <cstdlib>

#include "template.hu"

#define TILE_SZ_A 128
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B)

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

  /********************************************************************
  *
  * Compute C = A x B
  *   where A is a (m x k) matrix
  *   where B is a (k x n) matrix
  *   where C is a (m x n) matrix
  *
  * Use register and shared memory tiling and thread coarsening
  *
  * NOTE: A and C are column major, B is row major
  *
  ********************************************************************/

  // Macros for accessing flattened matrices
  #define A(row,col) A[(row) + (col)*m]
  #define B(row,col) B[(row)*n + (col)]
  #define C(row,col) C[(row) + (col)*m]

  // INSERT KERNEL CODE HERE

  // SSL Hint (9/6/21): try using just one register for the tile of A 
  // rather than several--in other words, load one value (per thread) 
  // from A and compute using that value rather than loading all values 
  // before doing the computation.  This approach seems to be slightly 
  // faster than the alternative.
	__shared__ float Bs[TILE_SZ_RATIO][TILE_SZ_B];
	int tid=threadIdx.x;
	int row=blockIdx.x*TILE_SZ_A+tid;
	int colstart=blockIdx.y*TILE_SZ_B;

	float sum[TILE_SZ_B] = {0};
	for(int i=0;i<k;i+=TILE_SZ_RATIO){
		int r = tid / TILE_SZ_B;
		int c = tid % TILE_SZ_B;
		Bs[r][c] = (i + r < k && colstart + c < n)
        ? B(i + r, colstart + c)
        : 0.0f;
		__syncthreads();

		for (int j = 0; j < TILE_SZ_RATIO; j++) {
			if (row < m && i + j < k) {
				float a = A(row, i + j);
				for (int c = 0; c < TILE_SZ_B; c++) {
					sum[c] += a * Bs[j][c];
				}
			}
		}
		__syncthreads();
	}
	for(int c = 0; c < TILE_SZ_B; c++) {
		if (row < m && colstart + c < n) {
			C(row, colstart + c) = sum[c];
		}
	}
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
	if ((transa != 'N') && (transa != 'n')) {
		printf("unsupported value of 'transa'\n");
		return;
	}

	if ((transb != 'T') && (transb != 't')) {
		printf("unsupported value of 'transb'\n");
		return;
	}

	if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
		printf("unsupported value of alpha\n");
		return;
	}

	if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
		printf("unsupported value of beta\n");
		return;
	}

    // Initialize thread block and kernel grid dimensions ---------------------

    // Your code need only consider the m, n, k, A, B, and C parameters of
    // the function, which provide the matrix sizes (m, n, k) and data
    // (A, B, C).

// Initialize thread block and kernel grid dimensions

	dim3 block(TILE_SZ_A, 1, 1);

	dim3 grid(
			(m + TILE_SZ_A - 1) / TILE_SZ_A,
			(n + TILE_SZ_B - 1) / TILE_SZ_B,
			1
	);

	// Invoke CUDA kernel

	mysgemm<<<grid, block>>>(m, n, k, A, B, C);

}

