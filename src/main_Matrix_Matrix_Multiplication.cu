#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdlib.h>
#include <math.h>
#include <stdio.h>


#define check_cuda_error(ans) { assert_cuda((ans), __FILE__, __LINE__); }
__host__ inline void assert_cuda(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


int const tile_width = 2;

__global__ void calc_mat_mat_multip_kernel(const int width, float* d_M, float* d_N, float* d_P)
{
	//int tile_width = blockDim.x; //=blockDim.y;
	//__shared__ float** Mds, ** Nds;
	//check_cuda_error(cudaMalloc((void**)&Mds, tile_width * sizeof(float)));
	//check_cuda_error(cudaMalloc((void**)&Nds, tile_width * sizeof(float)));

	__shared__ float Mds[tile_width][tile_width];
	__shared__ float Nds[tile_width][tile_width];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	//identify the row and colum of the d_p element to work on
	int row = by * tile_width + ty;
	int col = bx * tile_width + tx;

	float pvalue = 0;

	//loop over the d_M and d_N tiles required to compute d_p element
	for (int ph = 0; ph < width / tile_width; ph++)
	{
		Mds[ty][tx] = d_M[row * width + ph * tile_width + tx];
		Nds[ty][tx] = d_N[(ph * tile_width + ty) * width + col];
		__syncthreads();
		for (int k = 0; k < tile_width; k++)
		{
			pvalue += Mds[ty][k] * Nds[k][tx];
		}
		__syncthreads();
	}
	d_P[row * width + col] = pvalue;

}


 __host__ void calc_square_matrix_multiplication(const int width, float* h_A, float* h_B, float* h_C)
{
	float* d_A, * d_B, * d_C;

	// Choose which GPU to run on, change this on a multi-GPU system.
	check_cuda_error(cudaSetDevice(0));

	//allocate required memory in GPU
	int size = (width * width) * sizeof(float);
	check_cuda_error(cudaMalloc((void**)&d_A, size));
	check_cuda_error(cudaMalloc((void**)&d_B, size));
	check_cuda_error(cudaMalloc((void**)&d_C, size));

	// Copy data from host to GPU
	check_cuda_error(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	check_cuda_error(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));


	//// CUDA hardware specifications
	//	//int dev_count; 
	//	//cudaGetDeviceCount(&dev_count);

	//	//cudaDeviceProp dev_prop;
	//	//for (int i = 0; i < dev_count; i++)
	//	//{
	//		//cudaGetDeviceProperties(&dev_prop, i);
	//	//}

	//cudaDeviceProp dev_prop;
	//cudaGetDeviceProperties(&dev_prop, 0);

	//int max_threads_in_a_block = dev_prop.maxThreadsPerBlock;
	//int max_blockDim_x = dev_prop.maxThreadsDim[0];
	//int max_blockDim_y = dev_prop.maxThreadsDim[1];
	//int max_blockDim_z = dev_prop.maxThreadsDim[2];

	//int max_gridDim_x = dev_prop.maxGridSize[0];
	//int max_gridDim_y = dev_prop.maxGridSize[1];
	//int max_gridDim_z = dev_prop.maxGridSize[2];

	//int max_shared_memory_per_block = dev_prop.sharedMemPerBlock - dev_prop.reservedSharedMemPerBlock;
	//int max_shared_momory_capacity_to_store_float = max_shared_memory_per_block / sizeof(float);

	//// Execute kernel
	//int num_threads = dev_prop.warpSize;
	//int num_blocks = ceil(width / float(num_threads));

	//if (num_blocks > max_gridDim_x)
	//{
	//	num_blocks = max_gridDim_x;
	//	num_threads = dev_prop.warpSize * ceil(n / float(dev_prop.warpSize));
	//	if (num_threads > max_blockDim_x)
	//	{
	//		//you can do two things!
	//		//1- rewrite the code to let a thread handle multiple vector addition
	//		//2- loop over the kernels!
	//	}
	//}

	// Execute kernel
	int num_blocks_x = tile_width;
	int num_blocks_y = tile_width;
	int num_threads_x = width/tile_width;
	int num_threads_y = width/tile_width;

	dim3 dimGrid(num_blocks_x, num_blocks_y, 1);
	dim3 dimBlock(num_threads_x, num_threads_y, 1);

	calc_mat_mat_multip_kernel <<< dimGrid, dimBlock >>> (width, d_A, d_B, d_C);

	// Check for any errors launching the kernel
	check_cuda_error(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	check_cuda_error(cudaDeviceSynchronize());


	// Copy output vector from GPU buffer to host memory.
	check_cuda_error(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

	check_cuda_error(cudaFree(d_A));
	check_cuda_error(cudaFree(d_B));
	check_cuda_error(cudaFree(d_C));
}

int main()
{
	//== Memory allocation for h_M, h_N, and h_P ===//
	const int width = 4;

	float* h_A, * h_B, * h_C;

	int size = (width * width) * sizeof(float);
	h_A = (float*)malloc(size);
	h_B = (float*)malloc(size);
	h_C = (float*)malloc(size);
	//==============================================//

	//=== initialise two input matrices ===//
	for (int i = 0; i < width*width; i++)
	{
		h_A[i] = i+1;
		h_B[i] = i+1;
	}
	//=====================================//


	calc_square_matrix_multiplication(width, h_A, h_B, h_C);

	for (int i = 0; i < width * width; i++)
	{
		fprintf(stderr, "%f\n", h_C[i]);
	}

	return 0;
}