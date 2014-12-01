NVCC=nvcc

OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui
CUDA_INCLUDEPATH=/usr/local/cuda-6.5/include

NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64

GCC_OPTS=-O3 -Wall -Wextra -m64

SieveOfSundaram: main.cu sieve.cpp parallelSieve.cu
	$(NVCC) -I ${CUDA_INCLUDEPATH} -o SieveOfSundaram main.cu -L $(NVCC_OPTS)

#sieve.o: sieve.cpp
#	g++ -c sieve.cpp ${GCC_OPTS} -I ${CUDA_INCLUDEPATH}

#parallelSieve.o: parallelSieve.cu
#	${NVCC} -c parallelSieve.cu ${NVCC_OPTS}

clean:
	rm -f *.o
	rm SieveOfSundaram
