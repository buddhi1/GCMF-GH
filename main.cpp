#include "main.h"

#include "GH-CUDA/ghcuda.cpp"
int main(int argc, char *argv[])
{
    long totalPairNum;
    int *PPID_list, *QQID_list;
    coord_t *pBaseCoords, *pOverlaycoords;
    int *pBVNum, *pOVNum; 
    long *pBVPSNum, *pOVPSNum;

    spatialJoin(argc, argv, &PPID_list, &QQID_list, &totalPairNum, 
    &pBaseCoords, &pOverlaycoords, 
    &pBVNum, &pBVPSNum, &pOVNum, &pOVPSNum);

    // for(int i=0; i<totalPairNum; ++i){
    //     cout<<*(PPID_list+i)<<", "<<*(QQID_list+i)<<endl;
    // }
    
    // ghcuda(PPID_list, QQID_list, 30);

    return 0;
}
