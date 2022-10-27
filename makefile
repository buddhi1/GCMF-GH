GCC = g++

LIBB = 
LIBCUDA = -L/usr/local/cuda/lib64
LIBRA = 
CFLAGS = -m64 -O2 -Wall -c -std=c++11 -I/usr/local/include
DEBUG =

all: GH-CUDA/gppolyclip_mDIV512.o GH-CUDA/ghcuda.o 
	$(GCC) -O2 $(LFLAGS) -o GH-CUDA/program GH-CUDA/ghcuda.o GH-CUDA/gppolyclip_mDIV512.o $(LIBB) $(LIBRA) $(LIBCUDA) -lcudart

# count: gppolyclip_mDIV512_withcount.o ghcuda.o 
# 	$(GCC) -O2 $(LFLAGS) -o program ghcuda.o gppolyclip_mDIV512_withcount.o $(LIBB) $(LIBRA) $(LIBCUDA) -lcudart

# basic: gppolyclip.o ghcuda.o 
# 	# $(GCC) -O2 $(LFLAGS) -o program ghcuda.o gppolyclip.o 
# 	$(GCC) -O2 $(LFLAGS) -o program ghcuda.o gppolyclip.o $(LIBB) $(LIBRA) $(LIBCUDA) -lcudart

GH-CUDA/ghcuda.o: GH-CUDA/ghcuda.cpp
	$(GCC) $(DEBUG) $(CFLAGS) -c GH-CUDA/ghcuda.cpp

GH-CUDA/gppolyclip_mDIV512.o: GH-CUDA/gppolyclip_mDIV512.cu
	nvcc -Xptxas -O2 -o GH-CUDA/gppolyclip_mDIV512.o -c GH-CUDA/gppolyclip_mDIV512.cu

# gppolyclip_mDIV512_withcount.o: gppolyclip_mDIV512_withcount.cu
# 	nvcc -x cu -w -m64  -o gppolyclip_mDIV512_withcount.o -c gppolyclip_mDIV512_withcount.cu

clean:
	rm GH-CUDA/*.o GH-CUDA/program