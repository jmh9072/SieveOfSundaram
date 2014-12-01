#include <iostream>

#include "sieve.cpp"
#include "parallelSieve.cu"

#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <time.h>

using namespace std;

int main()
{
	int choice = -1;
	int bound = 0;
	
	int *primeArray, *findArray;
	int *cpuArray;
	
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
			
		cout << "What number should we find primes up to?" << endl;
		cin >> bound;
		
		cout << "Creating reference prime array...";
		int * goldArray = new int[bound + 1];
		sundaramSieve(bound, goldArray);
		cout << "done." << endl;
		
		const dim3 a_gridSize(bound / 1024, 1, 1);
		const dim3 a_blockSize(512, 1, 1);
		const dim3 b_gridSize(bound, 1, 1);
		const dim3 b_blockSize(32, 16, 1);

		const dim3 t_gridSize(bound,1,1);
		const dim3 t_blockSize(32,16,1);
		
		if (choice >= 3) //If we've run a GPU algorithm, allocate some memory
		{
			cout << "Allocating " << 2 * sizeof(int) * (2 * bound + 2) / 1024.0 / 1024.0 << "MB of memory" << endl;
			checkCudaErrors(cudaMalloc(&primeArray, sizeof(int) * (2*bound + 2)));
			checkCudaErrors(cudaMalloc(&findArray, sizeof(int) * (2*bound + 2)));
		}

		switch (choice)
		{
			case 0:
			{
				t = clock();
				for (int i = 0; i < 10000; i++)
				{
					cpuArray = new int[bound + 1];
					eratosthenesSieve(bound, cpuArray);
					break;
				}
			}
						
			case 1:
			{
				t = clock();
				for (int i = 0; i < 10000; i++)
				{
					cpuArray = new int[bound + 1];
					sundaramSieve(bound, cpuArray);
					break;
				}
			}
			case 2:
				//not yet implemented
			break;
			
			case 3:
				t = clock();
				//for (int i = 0; i < 10000; i++)
				{
					checkCudaErrors(cudaMemset(findArray, 0, sizeof(int) * (2*bound + 2)));
					checkCudaErrors(cudaMemset(primeArray, 1, sizeof(int) * (2*bound + 2)));
					sundPartOnePerRow<<<t_gridSize, t_blockSize>>>(bound, findArray);
					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
					sundPartTwoPerElementOneD<<<t_gridSize, t_blockSize>>>(bound, findArray, primeArray);
					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
				}
			break;
			
			case 4:
				t = clock();
				//for (int i = 0; i < 10000; i++)
				{
					checkCudaErrors(cudaMemset(findArray, 0, sizeof(int) * (2*bound + 2)));
					checkCudaErrors(cudaMemset(primeArray, 1, sizeof(int) * (2*bound + 2)));
					sundPartOnePerRow<<<t_gridSize, t_blockSize>>>(bound, findArray);
					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
					sundPartTwoPerElementTwoD<<<b_gridSize, b_blockSize>>>(bound, findArray, primeArray);
					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
				}
			break;
			
			case 5:
				t = clock();
				//for (int i = 0; i < 10000; i++)
				{
					checkCudaErrors(cudaMemset(findArray, 0, sizeof(int) * (2*bound + 2)));
					checkCudaErrors(cudaMemset(primeArray, 1, sizeof(int) * (2*bound + 2)));
					sundPartOnePerElement<<<t_gridSize, t_blockSize>>>(bound, findArray);
					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
					//sundPartTwoPerElementOneD<<<t_gridSize, t_blockSize>>>(bound, findArray, primeArray);
					//cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
				}
			break;
			
			case 6:
				t = clock();
				//for (int i = 0; i < 10000; i++)
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
		cout << "Time taken to run: " << (total_time / 100) << " sec\n" << endl;
		
		if (choice >= 3) //If we've run a GPU algorithm, copy then free the memory
		{
			int *validatePrimeArray = new int[bound + 1];
			checkCudaErrors(cudaMemcpy(validatePrimeArray, findArray, sizeof(int) * (bound + 1), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaFree(findArray));
			checkCudaErrors(cudaFree(primeArray));
			validatePrimes(bound, goldArray, validatePrimeArray);
			delete [] validatePrimeArray;
		}
		else
		{
			validatePrimes(bound, goldArray, cpuArray);
		}
		delete [] cpuArray;

	}
}
