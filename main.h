#include "GH-CUDA/ghcuda.cpp"

int spatialJoin(int argc, char* argv[], int **djxyVector_return, int **dPiPFlag_return, long *pairNum_return);
void CopyFromGPU(void** destinationData, void* sourceData, int dataSize, char isNew);