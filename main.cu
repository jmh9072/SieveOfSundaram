#include <iostream>

#include "sieve.cpp"
#include "parallelSieve.cu"

#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <time.h>

using namespace std;

int main(int argc, char* argv[])
{
	int choice = -1;
	int bound = 0;
	
	bool *primeArray, *findArray;
	
	clock_t t;
	float total_time;
	
	while (1)
	{
		cout << "Which algorithm would you like to run?" << endl;
		cout << "0. Sieve of Eratosthenes (serial)" << endl;
		cout << "1. Sieve of Sundaram (serial)" << endl;
		cout << "2. Sieve of Sundaram (serial, optimized)" << endl;
		cout << "3. Sieve of Sundaram (parallel, GPU)" << endl;
		cout << "4. Sieve of Sundaram (parallel, GPU)" << endl;
		cout << "5. Sieve of Sundaram (parallel, GPU)" << endl;
		cout << "6. Sieve of Sundaram (parallel, GPU)" << endl;
		cout << "7. Exit" << endl;
		cin >> choice;
		
		//Process exit
		if (choice == 7)
			return 0;
		
		if (choice < 0 || choice > 7)
			continue;
			
		cout << "What number should we find primes up to?";
		cin >> bound;
		
		const dim3 a_blockSize(1024, 1, 1);
		const dim3 a_gridSize(bound / 1024, 1, 1);
		const dim3 b_blockSize(32, 32, 1);
		const dim3 b_gridSize(bound / 1024 / 2, 1, 1);
		
		switch (choice)
		{
			case 0:
			{
				t = clock();
				for (int i = 0; i < 10000; i++)
				{
					bool * eratosArray = new bool[bound + 1];
					eratosthenesSieve(bound, eratosArray);
					break;
				}
			}
						
			case 1:
			{
				t = clock();
				for (int i = 0; i < 10000; i++)
				{
					bool * sundArray = new bool[bound + 1];
					sundaramSieve(bound, sundArray);
					break;
				}
			}
			case 2:
				//not yet implemented
			break;
			
			case 3:
				t = clock();
				for (int i = 0; i < 10000; i++)
				{
					checkCudaErrors(cudaMalloc(&primeArray, sizeof(bool) * (bound + 1)));
					checkCudaErrors(cudaMalloc(&findArray, sizeof(bool) * (bound + 1)));
					checkCudaErrors(cudaMemset(findArray, 0, sizeof(bool) * (bound + 1)));
					checkCudaErrors(cudaMemset(primeArray, 1, sizeof(bool) * (bound + 1)));
					sundPartOnePerRow<<<a_gridSize, a_blockSize>>>(bound, findArray);
					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
					sundPartTwoPerElementOneD<<<a_gridSize, a_blockSize>>>(bound, findArray, primeArray);
					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
				}
			break;
			
			case 4:
				t = clock();
				for (int i = 0; i < 10000; i++)
				{
					sundPartOnePerRow<<<a_gridSize, a_blockSize>>>(bound, findArray);
					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
					sundPartTwoPerElementTwoD<<<b_gridSize, b_blockSize>>>(bound, findArray, primeArray);
					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
				}
			break;
			
			case 5:
				t = clock();
				for (int i = 0; i < 10000; i++)
				{
					sundPartOnePerElement<<<b_gridSize, b_blockSize>>>(bound, findArray);
					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
					sundPartTwoPerElementOneD<<<a_gridSize, a_blockSize>>>(bound, findArray, primeArray);
					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
				}
			break;
			
			case 6:
				t = clock();
				for (int i = 0; i < 10000; i++)
				{
					sundPartOnePerElement<<<b_gridSize, b_blockSize>>>(bound, findArray);
					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
					sundPartTwoPerElementTwoD<<<b_gridSize, b_blockSize>>>(bound, findArray, primeArray);
					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
				}
			break;
			
			default:
			break;
		}
		t = clock() - t;
		total_time = ((float)t) / CLOCKS_PER_SEC;
		std::cout << "Time taken to run: " << (total_time / 100) << " sec\n";
		
		//bool *validatePrimeArray = new bool[bound + 1];
		//delete [] validatePrimeArray;
		//validatePrimes(bound, );
		
		//checkCudaErrors(cudaMemcpy(validatePrimeArray, primeArray, sizeof(bool) * (bound + 1), cudaMemcpyDeviceToHost));
		
		if (choice >= 3) //If we've run a GPU algorithm
		{
			checkCudaErrors(cudaFree(findArray));
			checkCudaErrors(cudaFree(primeArray));
		}
	}
	return 0;
}
