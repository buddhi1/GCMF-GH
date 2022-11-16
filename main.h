#include "GH-CUDA/ghcuda.cpp"
#include "Types.h"

int spatialJoin(int argc, char* argv[], int  **PPID_list, int **QQID_list, long *totalPairNum, 
    coord_t **pBaseCoords, coord_t **pOverlaycoords, 
    int **pBVNum, long **pBVPSNum, int **pOVNum, long **pOVPSNum);