// #include "GH-CUDA/ghcuda.cpp"
#include <iostream>
#include "Types.h"

int spatialJoin(int argc, char* argv[], int  **pIDList, int **qIDList, long *totalPairNum, 
    coord_t **pBaseCoords, coord_t **pOverlayCoords, 
    int **pBVNum, long **pBVPSNum, int **pOVNum, long **pOVPSNum);
int ghcuda(int pIDList[], int qIDList[], int totalNumPairs,
          /*coord_t *baseCoords, coord_t *overlayCoords, */
          coord_t *bCoords, coord_t *oCoords,
          int *pBVNum, long *pBVPSNum, int *pOVNum, long *pOVPSNum);
 