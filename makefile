GCC = g++

LIBB = 
LIBCUDA = -L/usr/local/cuda/lib64
LIBRA = 
CFLAGS = -m64 -O2 -Wall -c -std=c++11 -I/usr/local/include
DEBUG =

# all: main.o GH-CUDA/gppolyclip_mDIV512.o GH-CUDA/ghcuda.o 
# 	$(GCC) -O2 $(LFLAGS) -o program GH-CUDA/ghcuda.o GH-CUDA/gppolyclip_mDIV512.o main.o $(LIBB) $(LIBRA) $(LIBCUDA) -lcudart
all: GH-CUDA/gppolyclip_mDIV512.o spatialJoin3.o  main.o
	$(GCC) -O2 $(LFLAGS) -o program  GH-CUDA/gppolyclip_mDIV512.o spatialJoin3.o main.o $(LIBB) $(LIBRA) $(LIBCUDA) -lcudart

GH-CUDA/ghcuda.o: GH-CUDA/ghcuda.cpp
	$(GCC) $(DEBUG) $(CFLAGS) -c GH-CUDA/ghcuda.cpp -o GH-CUDA/ghcuda.o

GH-CUDA/gppolyclip_mDIV512.o: GH-CUDA/gppolyclip_mDIV512.cu
	nvcc -Xptxas -O2 -o GH-CUDA/gppolyclip_mDIV512.o -c GH-CUDA/gppolyclip_mDIV512.cu

spatialJoin3.o: spatialJoin3.cu
	nvcc -Xptxas -O2 -o spatialJoin3.o -c spatialJoin3.cu

main.o: main.cpp
	$(GCC) $(DEBUG) $(CFLAGS) -c main.cpp

clean:
	rm *.o
	rm GH-CUDA/*.o program