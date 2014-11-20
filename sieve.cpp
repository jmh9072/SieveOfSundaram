// ConsoleApplication1.cpp : Defines the entry point for the console application.
// TODO: Run a bunch of times and average runtimes

#include "stdafx.h"
#include "stdio.h"
#include "math.h"
#include "string.h"
#include <iostream>


/// This function performs a serial Sieve of Eratosthenes to find all prime number
void eratosthenesSieve(int bound, bool * primeArray)
{
	int sqrtBound = (int)sqrt((double)bound);
	memset(primeArray, 0, sizeof(bool) * (bound + 1));
	for (int m = 2; m <= sqrtBound; m++)
	{
		if (!primeArray[m])
		{
			for (int k = m * m; k <= bound; k += m)
			{
				primeArray[k] = true;
			}
		}
	}

	//this code block was used for debugging
	//for (int m = 2; m < bound; m++)
	//{
	//	if (!primeArray[m])
	//	{
	//		std::cout << m << ", ";
	//	}
	//}
}

///This function performs a serial Sieve of sundaram to find all primes
void sundaramSieve(int bound, bool * primeArray)
{
	bool* findArray = new bool[bound + 1];
	memset(findArray, 0, sizeof(bool) * (bound + 1)); 
	memset(primeArray, 1, sizeof(bool) * (bound + 1));

	for (int i = 1; i < bound; i++)
	{
		for (int j = i; j <= (bound - i) / (2 * i + 1); j++)
		{
			findArray[i + j + (2 * i * j)] = true; 
		}
	}
	for (int i = 1; i < ((bound - 1) / 2); i++)
	{
		if (!findArray[i])
		{
			primeArray[((i * 2) + 1)] = false; 
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

void sundPartOneSerial(int bound, bool * findArray)
{
	int max = 0; 
	int denom = 0; 
	for(int i = 1; i < bound; i++)
	{
		denom = (i << 1) + 1; 
		max = (bound - i) / denom; 
		for(int j = i; j <= max; j++)
		{
			findArray[i + j * denom] = true; 
		}
	}
}

void sundPartTwoSerial(int bound, bool * findArray, bool * primeArray)
{
	int max = (bound - 1) >> 1; 
	for(int i = 1; i < max; i++)
	{
		if(!findArray[i])
		{
			primeArray[((i << 1) + 1)] = false; 
		}
	}
	primeArray[2] = false; 
}

///This function compares two arrays to see if they match in the range of prime numbers
void validatePrimes(int bound, bool* goldArray, bool* checkArray)
{
	for (int i = 2; i <= bound; i++)
	{
		if (goldArray[i] != checkArray[i])
		{
			std::cout << "Difference at Position " << i << "\n";
			std::cout << "Array is Incorrect! \n";
			return;
		}
	}
	std::cout << "Found all primes! \n"; 
}

int main(int argc, char* argv[])
{
	int choice = -1;
	int bound = 0;
	
	bool *primeArray, *findArray;
	const dim3 a_blockSize(1024, 1, 1);
	const dim3 a_gridSize(bound / 1024, 1, 1);
	const dim3 b_blockSize(32, 32, 1);
	const dim3 b_gridSize(bound / 1024 / 2, 1, 1);
	
	while (True)
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
			return 0
		
		if (choice < 0 || choice > 4)
			continue;
			
		cout << "What number should we find primes up to?";
		cin >> bound;

		checkCudaErrors(cudaMalloc(&primeArray, sizeof(bool) * (bound + 1)));
		checkCudaErrors(cudaMalloc(&findArray, sizeof(bool) * (bound + 1)));
		checkCudaErrors(cudaMemSet(findArray, 0, sizeof(bool) * (bound + 1)));
		checkCudaErrors(cudaMemSet(primeArray, 1, sizeof(bool) * (bound + 1)));
		
		switch (choice)
		{
			case 0:
			bool * eratosArray = new bool[bound + 1];
			eratosthenesSieve(bound, eratosArray);
			break;
			
			case 1:
			bool * sundArray = new bool[bound + 1];
			sundaramSieve(bound, sundArray);
			break;
			
			case 2:
			//not yet implemented
			break;
			
			case 3:

			sundPartOnePerRow<<<a_gridSize, a_blockSize>>>(bound, findArray);
			cudaDeviceSyncronize(); checkCudaErrors(cudaGetLastError());
			sundPartTwoPerElementOneD<<<a_gridSize, a_blockSize>>>(bound, findArray, primeArray);
			cudaDeviceSyncronize(); checkCudaErrors(cudaGetLastError());
			break;
			
			case 4:
			sundPartOnePerRow<<<a_gridSize, a_blockSize>>>(bound, findArray);
			cudaDeviceSyncronize(); checkCudaErrors(cudaGetLastError());
			sundPartTwoPerElementTwoD<<<b_gridSize, b_blockSize>>>(bound, findArray, primeArray);
			cudaDeviceSyncronize(); checkCudaErrors(cudaGetLastError());
			break;
			
			case 5:
			sundPartOnePerElement<<<b_gridSize, b_blockSize>>>(bound, findArray);
			cudaDeviceSyncronize(); checkCudaErrors(cudaGetLastError());
			sundPartTwoPerElementOneD<<<a_gridSize, a_blockSize>>>(bound, findArray, primeArray);
			cudaDeviceSyncronize(); checkCudaErrors(cudaGetLastError());
			break;
			
			case 6:
			sundPartOnePerElement<<<b_gridSize, b_blockSize>>>(bound, findArray);
			cudaDeviceSyncronize(); checkCudaErrors(cudaGetLastError());
			sundPartTwoPerElementTwoD<<<b_gridSize, b_blockSize>>>(bound, findArray, primeArray);
			cudaDeviceSyncronize(); checkCudaErrors(cudaGetLastError());
			break;
			
			default:
			break;
		}
		bool *validatePrimeArray = new bool[bound + 1];
		cudaCheckErrors(cudaMemCpy(validatePrimeArray, primeArray, sizeof(bool) * (bound + 1), cudaMemCpyDeviceToHost));
		
		validatePrimes(bound, );
		
		checkCudaErrors(cudaFree(findArray));
		checkCudaErrors(cudaFree(primeArray));
	}
	return 0;
}



