#include "main.h"

// #include "GH-CUDA/ghcuda.cpp"

using namespace std;

int main(int argc, char *argv[])
{
    long totalPairNum;
    int *pIDList, *qIDList;
    coord_t *pBaseCoords, *pOverlayCoords;
    int *pBVNum, *pOVNum; 
    long *pBVPSNum, *pOVPSNum;

    spatialJoin(argc, argv, &pIDList, &qIDList, &totalPairNum, 
    &pBaseCoords, &pOverlayCoords, 
    &pBVNum, &pBVPSNum, &pOVNum, &pOVPSNum);

    //remove this line to intersect all candidate pairs. 
    // Defines a small pair set for debugging
    // totalPairNum=30; 

    // for(int i=0; i<totalPairNum; ++i){
    //     cout<<*(pIDList+i)<<", "<<*(qIDList+i)<<endl;
    // }
    // printf("\nPair print end (from main.cpp)\n");
    ghcuda(pIDList, qIDList, totalPairNum,
          pBaseCoords, pOverlayCoords, 
          pBVNum, pBVPSNum, pOVNum, pOVPSNum);
 

    return 0;
}
