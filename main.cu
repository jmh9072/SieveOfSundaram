#include <iostream>

#include "sieve.cpp"
#include "parallelSieve.cu"

using namespace std;

int main(int argc, char* argv[])
{
	int choice = -1;
	int bound = 0;
	
	bool *primeArray, *findArray;
	const dim3 a_blockSize(1024, 1, 1);
	const dim3 a_gridSize(bound / 1024, 1, 1);
	const dim3 b_blockSize(32, 32, 1);
	const dim3 b_gridSize(bound / 1024 / 2, 1, 1);
	
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
		
		if (choice < 0 || choice > 4)
			continue;
			
		cout << "What number should we find primes up to?";
		cin >> bound;

		checkCudaErrors(cudaMalloc(&primeArray, sizeof(bool) * (bound + 1)));
		checkCudaErrors(cudaMalloc(&findArray, sizeof(bool) * (bound + 1)));
		checkCudaErrors(cudaMemset(findArray, 0, sizeof(bool) * (bound + 1)));
		checkCudaErrors(cudaMemset(primeArray, 1, sizeof(bool) * (bound + 1)));
		
		switch (choice)
		{
			case 0:
			{
				bool * eratosArray = new bool[bound + 1];
				eratosthenesSieve(bound, eratosArray);
				break;
			}
						
			case 1:
			{
				bool * sundArray = new bool[bound + 1];
				sundaramSieve(bound, sundArray);
				break;
			}
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
		checkCudaErrors(cudaMemcpy(validatePrimeArray, primeArray, sizeof(bool) * (bound + 1), cudaMemcpyDeviceToHost));
		
		//validatePrimes(bound, );
		
		checkCudaErrors(cudaFree(findArray));
		checkCudaErrors(cudaFree(primeArray));
	}
	return 0;
}
