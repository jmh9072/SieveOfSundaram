// ConsoleApplication1.cpp : Defines the entry point for the console application.
// TODO: Run a bunch of times and average runtimes

#include "stdio.h"
#include "math.h"
#include "string.h"
#include <iostream>

using namespace std;

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
	for (int i = 0; i <= 20; i++)
	{
		string s1, s2;
		if (goldArray[i])
			s1 = "compos.";
		else
			s1 = "prime";
		if (checkArray[i])
			s2 = "compos.";
		else
			s2 = "prime";
		cout << i << ".\t" << s1 << "\t" << s2 << endl;
	}
		
	for (int i = 0; i <= bound; i++)
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




