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
	
	bool *primeArray, *findArray;
	bool *cpuArray;
	
	clock_t t;
	float total_time;
	
	while (1)
	{
		cout << "Which algorithm would you like to run?" << endl;
		cout << "0. Sieve of Eratosthenes (serial)" << endl;
		cout << "1. Sieve of Sundaram (serial)" << endl;
		//cout << "2. Sieve of Sundaram (serial, optimized)" << endl;
		cout << "3. Sieve of Sundaram (GPU - PerRow, 1D)" << endl;
		cout << "4. Sieve of Sundaram (GPU - PerRow, 2D)" << endl;
		cout << "5. Sieve of Sundaram (GPU - PerElement, 1D)" << endl;
		cout << "6. Sieve of Sundaram (GPU - PerRow, 2D)" << endl;
		cout << "7. Sieve of Eratosthenes (GPU - Per Element 1D)" << endl;
		cout << "8. Sieve of Eratosthenes (GPU - Per Element 2D)" << endl;
		cout << "9. Exit" << endl;
		cin >> choice;
		
		//Process exit
		if (choice == 9)
			return 0;
		
		if (choice < 0 || choice > 9)
			continue;
			
		cout << "What number should we find primes up to?" << endl;
		cin >> bound;
		
		cout << "Creating reference prime array...";
		bool * goldArray = new bool[bound + 1];
		sundaramSieve(bound, goldArray);
		cout << "done." << endl;
		
		const dim3 a_gridSize(bound / 1024, 1, 1);
		const dim3 a_blockSize(512, 1, 1);
		const dim3 b_gridSize(bound / (32*32) / 2, bound / (32*32) / 2, 1);
		const dim3 b_blockSize(32, 32, 1);
		
		const dim3 c_gridSize(bound,1,1);
		const dim3 c_blockSize(512, 1, 1);

		const dim3 t_gridSize(bound / (32*32),1,1);
		const dim3 t_blockSize(1024,1,1);
		
		if (choice >= 3) //If we've run a GPU algorithm, allocate some memory
		{
			cout << "Allocating " << 2 * sizeof(bool) * (bound + 1) / 1024.0 / 1024.0 << "MB of memory" << endl;
			checkCudaErrors(cudaMalloc(&primeArray, sizeof(bool) * (bound +1)));
			checkCudaErrors(cudaMalloc(&findArray, sizeof(bool) * (bound + 1)));
			checkCudaErrors(cudaMemset(findArray, 0, sizeof(bool) * (bound + 1)));
			checkCudaErrors(cudaMemset(primeArray, 1, sizeof(bool) * (bound + 1)));
		}

		switch (choice)
		{
			case 0:
			{
				t = clock();
				//for (int i = 0; i < 10000; i++)
				{
					cpuArray = new bool[bound + 1];
					eratosthenesSieve(bound, cpuArray);
					break;
				}
			}
						
			case 1:
			{
				t = clock();
				//for (int i = 0; i < 10000; i++)
				{
					cpuArray = new bool[bound + 1];
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
					cout << "part one" << endl;
					sundPartOnePerElement<<<b_gridSize, b_blockSize>>>(bound, findArray);
					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
					cout << "part two" << endl;
					sundPartTwoPerElementOneD<<<t_gridSize, t_blockSize>>>(bound, findArray, primeArray);
					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
					cout << "done" << endl;
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
			
			case 7:
				t = clock();
				//for (int i = 0; i < 10000; i++)
				{
					checkCudaErrors(cudaMemset(primeArray, 0, sizeof(bool) * (bound + 1)));
					eratosPerElement<<<c_gridSize, c_blockSize>>>(bound, primeArray);
					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
				}
			break;
			case 8:
				t = clock();
				//for (int i = 0; i < 10000; i++)
				{
					checkCudaErrors(cudaMemset(primeArray, 0, sizeof(bool) * (bound + 1)));
					eratosPerElement2D<<<b_gridSize, b_blockSize>>>(bound, primeArray);
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
			bool *validatePrimeArray = new bool[bound + 1];
			checkCudaErrors(cudaMemcpy(validatePrimeArray, primeArray, sizeof(bool) * (bound + 1), cudaMemcpyDeviceToHost));
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
		return 0;
	}
}
