#include <cuda.h>
#include <cuda_runtime.h>

///all parallel implementations of this algorithim will require two functions or else delay a function significantly

__global__
void sundPartOnePerRow(int bound, bool * findArray)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	
	if(idx < 1)
	{
	return;
	}

	if (idx > bound)
		return;
	
	int denom = (idx * 2) + 1; 
	int max = (bound - idx) / denom; 
	
	for(int j = idx; j <= max; j++)
	{
		findArray[idx + j * denom] = true; 
	}
}

__global__
void sundPartOnePerElement(int bound, bool * findArray)
{
	int idx = 1024 * blockIdx.x + threadIdx.x; //x is j
	
	if(idx == 0) //j >= 1 
	{
		return;
	} 
	
	if(idx >= bound) //j < bound 
	{
		return; 
	}
	
	int idy = 1024 * blockIdx.y + threadIdx.y; //y is i
	
	if(idy == 0) //i >= 1
	{
		return;
	}
	if(idy >= bound) // i < bound
	{
		return; 
	}
	if(idy > idx) //i <= j
	{
		return;
	}
	
	int bin = idy + idx + ((idy * idx) << 1); //form i + j + 2ij might be better to do parts of this function individually
	
	if( bin > bound) // i + j + 2ij <= bound
	{
		return;
	}
	
	findArray[bin] = true; //collisions arnt a problem as its a set 
	
}

__global__
void sundPartTwoPerElementOneD(int bound, bool * findArray, bool * primeArray)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	if(idx == 0) //let thread 0 handle setting 2 as prime 
	{
		primeArray[2] = false; 
		return;
	}
	int realBound = (bound - 1) >> 1;

	if(idx >= realBound)
	{
		return;
	}
	
	if(!findArray[idx])
	{
		int bin = (idx << 1) + 1;
		primeArray[bin] = false; 
	}
}

__global__
void sundPartTwoPerElementTwoD(int bound, bool * findArray, bool * primeArray)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	int idy = blockDim.y * blockIdx.y + threadIdx.y; 
	int id = idx + idy;

	if(id == 0) //let thread 0 handle setting 2 as prime 
	{
		primeArray[2] = false; 
		return;
	}
	int realBound = (bound - 1) >> 1;

	if(id >= realBound)
	{
		return;
	}
	
	if(!findArray[id])
	{
		int bin = (id << 1) + 1;
		primeArray[bin] = false; 
	}
}

__global__
void eratosPerElement(int bound, bool * primeArray)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int sqrtBound = (int)sqrt((double)bound); 
	if (id == 0)
	{
		primeArray[0] = true;
		primeArray[1] = true;
	}
	if(id < 2)
	{
		return; 
	}
	if(id > sqrtBound)
	{
		return; 
	}
	for(int k = id * id; k <= bound; k+=id)
	{
		primeArray[k] = true; 
	}
}


///this parallel function should be launched in the following manner
///for( int i = 2; i < (bound / 2); i++)
///{
///		if(!primeArray[i])
///		{
///			eratosParallelMult<<<(bound /2)/1024, 1024>>>(i, bound, primeArray); //or some other way to calculate size dynamically
///		}
///}
__global__
void eratosParallelMult(int i, bool bound, bool * primeArray)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	
	int bin = i * idx; 
	
	if(bin > bound)
	{
		return;
	}
	
	primeArray[bin] = true;
}
///this should work because we dont care about collisions and all eratos does is find multiples, this will do some redundant calculationg but hopefully so fast it doesnt matter
__global__
void eratosPerElement2D(int bound, bool * primeArray)
{
	int sqrtBound = (int)sqrt((double)bound); 
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	if (idx == 0)
	{
		primeArray[0] = true;
		primeArray[1] = true;
	}
	if(idx < 2)
	{
		return; 
	}
	int idy = blockDim.y * blockIdx.y + threadIdx.y; 
	if(idy < 2)
	{
		return; 
	}
	int bin = idx * idy; 
	if(bin > bound)
	{
		return; 
	}
	primeArray[bin] = true;
}
