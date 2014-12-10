#include <cuda.h>
#include <cuda_runtime.h>

__global__
void sundPartOnePerElement(int bound, bool * findArray)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; //x is j
	
	if (idx == 0) //j >= 1 
	{
		return;
	} 
	
	if (idx >= bound) //j < bound 
	{
		return; 
	}
	
	int idy = blockDim.x * blockIdx.x + threadIdx.y; //y is i
	
	if (idy == 0) //i >= 1
	{
		return;
	}
	if (idy >= bound) // i < bound
	{
		return; 
	}
	if (idy > idx) //i <= j
	{
		return;
	}
	
	int bin = idy + idx + ((idy * idx) << 1); //form i + j + 2ij might be better to do parts of this function individually
	
	if (bin > bound) // i + j + 2ij <= bound
	{
		return;
	}
	
	findArray[bin] = true; //collisions arnt a problem as its a set 
}

int main()
{
	int bound = 10000000;
	
	const dim3 gridSize(bound/1024, 1, 1);
	const dim3 blockSize(32, 32, 1);

	checkCudaErrors(cudaMalloc(&findArray, sizeof(bool) * (bound + 1)));
	checkCudaErrors(cudaMemset(findArray, 0, sizeof(bool) * (bound + 1)));

	sundPartOnePerElement<<<gridSize, blockSize>>>(bound, findArray);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}