#include <cuda.h>
#include <cuda_runtime.h>

///This function performs a serial Sieve of sundaram to find all primes
void sundaramSieve(int bound, bool * primeArray)
{
	bool* findArray = new bool[bound + 1];
	memset(findArray, 0, sizeof(bool) * (bound + 1)); 
	memset(primeArray, 1, sizeof(bool) * (bound + 1));
	int max = 0; 
	int denom = 0;

	for (int i = 1; i < bound; i++)
	{
		denom = (i << 1) + 1; 
		max = (bound - i) / denom; 
		for (int j = i; j <= max; j++)
		{
			findArray[i + j * denom] = true; 
		}
	}
	for (int i = 1; i < ((bound - 1) / 2); i++)
	{
		if (!findArray[i])
		{
			primeArray[((i << 1) + 1)] = false; 
		}
	}
	primeArray[2] = false; // Sundaram doesnt find two so ill find it for it

	//code block below used for debuggin purposes
	//for (int m = 2; m < bound; m++)
	//{
	//	if (!primeArray[m])
	//	{
	//		std::cout << m << ", ";
	//	}
	//}
}

///all parallel implementations of this algorithim will require two functions or else delay a function significantly

__global__
void sundPartOnePerRow(int bound, bool * findArray)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	if(idx >= bound)
	{
		return;
	}
	int denom = (idx << 1) + 1; 
	int max = (bound - idx) >> 1; 
	for(int j = idx; j <= max; j++)
	{
		findArray[idx + j * denom] = true; 
	}
}

__global__
void sundPartOnePerElement(int bound, bool * findArray)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; //x is j
	
	if(idx == 0) //j >= 1 
	{
		return;
	}
	
	if(idx >= bound) //j < bound 
	{
		return; 
	}
	
	int idy = blockDim.y * blockIdx.y + threadIdx.y; //y is i
	
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
	if(idx < 2)
	{
		return; 
	}
	if(idx == 2) //let thread 0 handle setting 2 as prime 
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

void sundPartTwoPerElementTwoD(int bound, bool * findArray, bool * primeArray)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	int idy = blockDim.y * blockIdx.y + threadIdx.y; 
	int id = idx + idy;
	if(id < 2)
	{
		return; 
	}
	if(id == 2) //let thread 0 handle setting 2 as prime 
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
	
	if(id < 2)
	{
		return; 
	}
	if(id > sqrtBound)
	{
		return; 
	}
	for(int k = id * id; k <= bound; k++)
	{
		primeArray[k] = true; 
	}
}