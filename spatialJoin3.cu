#include <stdio.h>
#include "GPU_Manage.h"
// #include "Types.h"
#include "Constants.h"
#include "GPU_Test.h"
#include "GPU_MBR.h"
#include "IO.h"
#include "SEQ_Overlay.h"
#include "GPU_Utility.h"
#include "Data_Visualization.h"
#include "Join.h"

// #include "GH-CUDA/ghcuda.cpp"
// #include "GH-CUDA/lib/polyclip.cpp"

cudaEvent_t start_GPU, stop_GPU;

void copyPolygon(
    double **polyPX, double **polyPY, double **polyQX, double **polyQY, 
    int *bVNum, long *bVPSNum, coord_t* baseCoords, 
    int *oVNum, long *oVPSNum,coord_t* overlayCoords,
    int baseID, int overlayID) {
    int sizeP=bVNum[baseID], sizeQ=oVNum[overlayID];    
    *polyPX=(double *)malloc(sizeP*sizeof(double));
    *polyPY=(double *)malloc(sizeP*sizeof(double));
    *polyQX=(double *)malloc(sizeQ*sizeof(double));
    *polyQY=(double *)malloc(sizeQ*sizeof(double));
    printf("size %d %d %d %d\n", bVPSNum[0], bVPSNum[1], bVNum[0], bVNum[1]);
    for(int j=0, i=bVPSNum[baseID]-bVNum[baseID]; i<sizeP; j++, i+=2){
        *(*polyPX+j)=baseCoords[i];
        *(*polyPY+j)=baseCoords[i+1];
    }
    for(int j=0, i=oVPSNum[baseID]-oVNum[baseID]; i<sizeP; j++, i+=2){
        *(*polyPX+j)=overlayCoords[i];
        *(*polyPY+j)=overlayCoords[i+1];
    }
}

// int main(int argc, char* argv[]){  
int spatialJoin(int argc, char* argv[], int  **pIDList, int **qIDList, long *totalPairNum, 
    coord_t **pBaseCoords, coord_t **pOverlayCoords, 
    int **pBVNum, long **pBVPSNum, int **pOVNum, long **pOVPSNum){  

    float Join_Total_Time_SEQ=0, Join_Total_Time_GPU=0;
    cudaError_t cudaMemError;
//------------------------ Console Input ---------------------------------- 
/*
First user input: dimSort
	1: Sorting just based on one dimension (default is X)
	0: Sorting based on both X and Y dimensions
Second user input: dimSelect
	If dimSort=1, this argument define which dimension should be picked for sorting (Values could be 'X' or 'Y')
*/
    int dimSort=1, dimSelect=1;
    if(argc<2){
       dimSort=1;
       dimSelect=0;
    }
    else if(argc<3){
      if(argv[1][0]=='2')dimSort=2;
      else dimSort=1;
      dimSelect=0;
    }
    else if(argc<4){
      if(argv[2][0]=='y')dimSelect=1;
      else dimSelect=0;
      if(argv[1][0]=='2'){dimSort=2;dimSelect=0;}
      else dimSort=1;
    }
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//=============================== SEQUENTIAL RUN ===============================
//------------------------------------------------------------------------------    
//------------------------------------------------------------------------------    
    coord_t* baseCoords=(coord_t*)malloc(MAX_POLYS_BASE*4*AVG_VERTEX_PER_BASE_POL*sizeof(coord_t));
    coord_t* overlayCoords=(coord_t*)malloc(MAX_POLYS_OVERLAY*4*AVG_VERTEX_PER_OVERLAY_POL*sizeof(coord_t));
    int *bVNum=(int*)malloc(sizeof(int)*MAX_POLYS_BASE);
    int *oVNum=(int*)malloc(sizeof(int)*MAX_POLYS_OVERLAY);
    long *bVPSNum=(long*)malloc(sizeof(long)*MAX_POLYS_BASE);
    long *oVPSNum=(long*)malloc(sizeof(long)*MAX_POLYS_OVERLAY);
    long bPolNum, oPolNum, bVNumSum=0, oVNumSum=0;    
    mbr_t* seqMBR=(mbr_t*)malloc(MAX_POLYS_BASE*4*sizeof(mbr_t));
    mbr_t* seqOMBR=(mbr_t*)malloc(MAX_POLYS_OVERLAY*4*sizeof(mbr_t));
    coord_t* seqMBR2=(coord_t*)malloc(MAX_POLYS_BASE*4*sizeof(coord_t));
    coord_t* seqOMBR2=(coord_t*)malloc(MAX_POLYS_OVERLAY*4*sizeof(coord_t));

    //=================== Reading First(base) Polygon ==========================

    char baseFileName[100], overlayFileName[100];
    switch(DATASET){
       case 1:
        //  strcpy(baseFileName, "/pylon5/cc560kp/danialll/Text_Datasets/admin_states.txt");
	    //  strcpy(overlayFileName, "/pylon5/cc560kp/danialll/Text_Datasets/urban_areas.txt");
        
        //  printf("\nDataset: admin - urban\n");
        //  strcpy(baseFileName, "../datasets/datasets/admin_states.txt");
	    //  strcpy(overlayFileName, "../datasets/datasets/urban_areas.txt");

         printf("Dataset: Ocean - Land\n");
         strcpy(baseFileName, "../datasets/datasets/ne_10m_ocean.txt");
	     strcpy(overlayFileName, "../datasets/datasets/ne_10m_land.txt");
        //  strcpy(overlayFileName, "../datasets/datasets/ne_10m_ocean.txt");
	    //  strcpy(baseFileName, "../datasets/datasets/ne_10m_land.txt");
         break;
       case 2:
         strcpy(baseFileName, "/pylon5/cc560kp/danialll/Text_Datasets/bases_242.txt");
         strcpy(overlayFileName, "/pylon5/cc560kp/danialll/Text_Datasets/overlay_300.txt");
         printf("\nDataset: bases - overlay\n");
         break;
       case 3:
         strcpy(baseFileName, "/pylon5/cc560kp/danialll/Text_Datasets/block_boundaries.txt");
         strcpy(overlayFileName, "/pylon5/cc560kp/danialll/Text_Datasets/water_bodies.txt");
         printf("\nDataset: boundaries - water\n");
         break;
       case 4:
         strcpy(baseFileName, "/pylon5/cc560kp/danialll/Text_Datasets/postal.txt");
         strcpy(overlayFileName, "/pylon5/cc560kp/danialll/Text_Datasets/sports.txt");
         printf("\nDataset: postal - sports\n");
         break;
    }
    bPolNum=ReadTextFormatPolygon2WithVector(baseFileName,bVNum, bVPSNum, seqMBR, seqMBR2, baseCoords, &bVNumSum, 1, MAX_POLYS_BASE, pPolygons);    
    printf("\n%lu Polygons with %lu vertices in total.\n",bPolNum,bVNumSum);
    oPolNum=ReadTextFormatPolygon2WithVector(overlayFileName, oVNum, oVPSNum, seqOMBR, seqOMBR2, overlayCoords, &oVNumSum, 1, MAX_POLYS_OVERLAY, qPolygons); 
    printf("\n%lu Polygons with %lu vertices in total.\n",oPolNum,oVNumSum);

    //==========================================================================
    printf("\npPolygons size %d qPolygons size %d\n", pPolygons.size(), qPolygons.size());
// PrintPolygon(baseCoords+2*bVPSNum[1485], bVNum[1486]);
PrintPolygon(baseCoords, 5);
printf("\n\n\n");
// PrintPolygon(overlayCoords+2*oVPSNum[10], oVNum[11]);
PrintPolygon(overlayCoords, 5);
//return;

printf("size %d %d %d %d %d\n", bVNum[0], bVNum[1], bVNum[2], bVPSNum[0], bVPSNum[1]);
// PrintPolygon(baseCoords+2*bVPSNum[0], bVNum[1]);

// =======================**************========================================
/* CPU data structures
----------------------------------------------------
bPolNum: total # polygons in the base layer
bVNumSum: sum of vertices in the base layer

bVNum[]: size of each polygon
bVPSNum[]: prefixsum of bVNum[]

baseCoords[]:coordinates of base polygon {x_i, y_i} pairs in the same array

oPolNum: total # polygons in the overlay layer
oVNumSum: sum of vertices in the overlay layer

oVNum[]: size of each polygon
oVPSNum[]: prefixsum of bVNum[]

overlayCoords[]:coordinates of overlay polygon {x_i, y_i} pairs in the same array
*/
// =======================**************========================================

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//================================== GPU RUN ===================================
//------------------------------------------------------------------------------    
//------------------------------------------------------------------------------    

// =======================**************========================================
/*CPU GPU data structures
----------------------------------------------------
bVNum[] dbVNum[]: size of each polygon
bVPSNum[] dbVPSNum[]: prefixsum of bVNum[]

baseCoords[] bCoords[]:coordinates of base polygon {x_i, y_i} pairs in the same array


oVNum[] doVNum[]: size of each polygon
oVPSNum[] doVPSNum[]: prefixsum of bVNum[]

overlayCoords[] oCoords[]:coordinates of overlay polygon {x_i, y_i} pairs in the same array
*/
// =======================**************========================================


//=========================== Reseting GPU device ==============================
    cudaError_t error_reset=cudaDeviceReset();    
    if(error_reset!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error_reset) );
       exit(-1);
    }
    cudaThreadExit();
    //size_t mem_free_0,mem_total_0;
    //cudaMemGetInfo  (&mem_free_0, &mem_total_0);
    //printf("\nFree: %lu  , Total: %lu\n",mem_free_0,mem_total_0);
    
    //==================== Running Kernel (CreateMBR) =========================
    //====================== Transfering data to GPU ==========================
    StartTimer(&start_GPU, &stop_GPU);
   
    int *dbVNum, *doVNum;
    coord_t *oCoords, *bCoords;
    mbr_t *doMBR, *dbMBR;
    long *dbVPSNum, *doVPSNum;

    //----------- Transfering polygon number variables to GPU ---------------
    CopyToGPU((void**)&dbVNum, bVNum, sizeof(int)*bPolNum, "dbVNum", 1);
    CopyToGPU((void**)&doVNum, oVNum, sizeof(int)*oPolNum, "doVNum", 1);
    CopyToGPU((void**)&dbVPSNum, bVPSNum, sizeof(long)*bPolNum, "dbVPSNum", 1);
    CopyToGPU((void**)&doVPSNum, oVPSNum, sizeof(long)*oPolNum, "doVPSNum", 1);
    //-----------------------------------------------------------------------
    //------------- Transfering polygon coordinates to GPU i-----------------
    CopyToGPU((void**)&bCoords, baseCoords, sizeof(coord_t)*2*bVNumSum, "bCoords", 1);
    CopyToGPU((void**)&oCoords, overlayCoords, sizeof(coord_t)*2*oVNumSum, "oCoords", 1);
    //-----------------------------------------------------------------------
    //----------------------- Transfering MBRs to GPU -----------------------
    CopyToGPU((void**)&dbMBR, seqMBR, 4*sizeof(mbr_t)*bPolNum, "dbMBR", 1);
    CopyToGPU((void**)&doMBR, seqOMBR, 4*sizeof(mbr_t)*oPolNum, "doMBR", 1);
    //-----------------------------------------------------------------------
    GPUSync("Transfering data to GPU");

    float runningTime_GPU_TransferData;
    Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_TransferData);
    printf("\n\nGPU running time for transfering data to GPU: %f (%f)\n",runningTime_GPU_TransferData, Join_Total_Time_GPU);
//==============================================================================

//GPUPrintVector(4*oPolNum, doMBR2, 0);
//SEQMBROverlap(bPolNum, oPolNum, seqMBR, seqOMBR, -1);
//return;


   /*

   SpatialJoin(int* bPolNum, int* oPolNum, int* bVNum, int* oVNum, coord_t* bCoord, coord_t* oCoord, coord_t * bMBR, coord_t* oMBR, mbr_t* bMBR2, mbr_t* oMBR2, int** jxyVector, int* pairNum);
  
   Input parameters:
   bPolNum: Number of polygons in layer 1
   bMBR: MBRs from polygons of coord_t type (float). Format (x1, y1, x2, y2) in a vector structure of size 4*bPolNum*sizeof(coord_t)
   bMBR2: MBRs from polygons of mbr_t type (long long). Same format as bMBR.
   bVNum: Number of vertices in each polygon. We have bPolNum polygons then bVNum has bPolNum int elements.
   bCoord: Vertices of all the polygons of layer b. Format (x1_0,y1_0, x2_0, y2_0,...., x1_2, y1_2,.....,x1_(bPolNum-1), y1_(bPolNum-1) )

   Output parameters:
   pairNum: Number of output pairs
   jxyVector: Output pairs in format (i1, j1, i2, j2,....) i1: index of polygon from b layer and j1 index of polygon from o layer.
   */


//--------------------------- Find Overlaping MBRs (novel approach) ---------------------------
    StartTimer(&start_GPU, &stop_GPU);

    int *djxyCounter, *djxyVector, polNum=bPolNum+oPolNum; 
    // int *djxyCounter, polNum=bPolNum+oPolNum; 
    cudaMemError=cudaMalloc((void**)&djxyCounter,sizeof(int)*(polNum));

    long pairNum=SortBaseMBROverlap(bPolNum, oPolNum, dbMBR, doMBR, &djxyCounter, &djxyVector, dimSort, dimSelect);
   
    printf("\n\n\tPolygon pairs candidate: %ld\n", pairNum);
    float runningTime_GPU_overlap2;
    Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_overlap2);
    printf("\nGPU Running Time For Computing MBR intersection (new approach %dD [dim:%c] ): %f (%f)\n",dimSort, 'X', runningTime_GPU_overlap2, Join_Total_Time_GPU);
    cudaFree(doMBR);
    cudaFree(dbMBR);
    cudaFree(djxyCounter);

//------------------------------------------------------------------------------

//GPUPrintVector(2*pairNum, djxyVector, 0);
//return;



//--------------------------- CMF filter for Polygon Test operation --------------------------
    StartTimer(&start_GPU, &stop_GPU);
    int *djxy2IndexList, *djPiPIndexList, *dPiPFlag, *djoinFlag;
    // int *djxy2IndexList, *djPiPIndexList, *djoinFlag;
    char* dPiPType;
    long eiNum, pairNum3, pipNum, workLoadNum;
    coord_t *dcMBR, *dbMBR2, *doMBR2;
    CopyToGPU((void**)&doMBR2, seqOMBR2, sizeof(coord_t)*oPolNum*4, "doMBR2", 1);
    CopyToGPU((void**)&dbMBR2, seqMBR2, sizeof(coord_t)*bPolNum*4, "dbMBR2", 1);

    GetCMBR(pairNum, djxyVector, dbMBR2, doMBR2, &dcMBR, &djPiPIndexList, &dPiPFlag, &dPiPType, &djoinFlag, &pipNum);

    float runningTime_GPU_PiPCMF;
    Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_PiPCMF);
    printf("\nGPU Running Time for CMF Filter for Point in Polygon Test: %f (%f)\n", runningTime_GPU_PiPCMF, Join_Total_Time_GPU);
//------------------------------------------------------------------------------

//--------------------------- Point in Polygon Test operation --------------------------
    StartTimer(&start_GPU, &stop_GPU);
    long wNum;

    wNum=PointInPolygonTest(bCoords, oCoords, pairNum, pipNum, djxyVector, djPiPIndexList, dPiPType, dbVPSNum, doVPSNum, dPiPFlag, djoinFlag);
    
    printf("\n\tNumber of within pairs: %ld\n", wNum);

    //PrintPairs(djxyVector, dPiPFlag, pairNum);
//GPUPrintVector(pairNum2, dEdgeIntersectCounter, 1);

    float runningTime_GPU_PiP;
    Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_PiP);
    printf("\nGPU Running Time for Point in Polygon Test: %f (%f)\n", runningTime_GPU_PiP, Join_Total_Time_GPU);
 
    
    /*StartTimer(&start_GPU, &stop_GPU);
    float runningTime_SEQ_PiP;
    long wNum2=SEQ_PointInPolygonTest(baseCoords, overlayCoords, pairNum, pipNum, djxyVector, djPiPIndexList, dPiPType, bVPSNum, oVPSNum, dPiPFlag, djoinFlag);
    printf("\n\tNumber of within pairs (Sequential): %ld\n", wNum2);
    StopTimer(&start_GPU, &stop_GPU, &runningTime_SEQ_PiP);
    printf("\n\tSequential Running Time for Point in Polygon Test: %f\n", runningTime_SEQ_PiP);
    return;*/
//------------------------------------------------------------------------------



//--------------------------- Applying Common MBR Filtering (novel approach) ---------------------------
    StartTimer(&start_GPU, &stop_GPU);
    poly_size_t *dbEdgeList, *doEdgeList;
    long *dbEdgePSCounter, *doEdgePSCounter, *dWorkLoadPSCounter;
//GPUPrintVector(4*oPolNum, doMBR2, 0);
//return;


    CountCMF(bCoords, oCoords, pairNum, djxyVector, djoinFlag, dbVNum, doVNum, dbVPSNum, doVPSNum, dcMBR, &dbEdgePSCounter, &doEdgePSCounter, &dWorkLoadPSCounter, &djxy2IndexList, &dbEdgeList, &doEdgeList, &eiNum, &workLoadNum);

    //printf("\n\tPolygon pair candidates after Applying CMF filter: %ld\n", eiNum);
    float runningTime_GPU_CCMF;
    Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_CCMF);
    printf("\nGPU Running Time for Counting Common MBR Filter: %f (%f)\n", runningTime_GPU_CCMF, Join_Total_Time_GPU);

    StartTimer(&start_GPU, &stop_GPU);

    ApplyCMF(bCoords, oCoords, pairNum, djxyVector, eiNum, djxy2IndexList, dbVNum, doVNum, dbVPSNum, doVPSNum, dcMBR, dbEdgePSCounter, doEdgePSCounter, dbEdgeList, doEdgeList);
    //GPUPrintVector(pairNum2*2, djxy2IndexList, 0);
    //GPUPrintVector(pairNum*2, djxyVector, 0);
    //GPUPrefixsumTest(dbEdgeCounter, dbEdgePSCounter, pairNum, 1);
    //GPUPrefixsumTest(doEdgeCounter, doEdgePSCounter, pairNum, 1);

    float runningTime_GPU_ACMF;
    Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_ACMF);
    printf("\nGPU Running Time for Applying Common MBR Filter: %f (%f)\n", runningTime_GPU_ACMF, Join_Total_Time_GPU);
    cudaFree(dcMBR);

//------------------------------------------------------------------------------

//--------------------------- Join/Overlay operations --------------------------
    StartTimer(&start_GPU, &stop_GPU);
    int* dSegmentIntersectJoinFlag;
    pairNum3=SegmentIntersectJoin(bCoords, oCoords, eiNum, djxyVector, djxy2IndexList, dbVPSNum, doVPSNum, dbEdgePSCounter, doEdgePSCounter, dbEdgeList, doEdgeList, &dSegmentIntersectJoinFlag);
    //pairNum3=SegmentIntersectJoin2(bCoords, oCoords, eiNum, djxyVector, djxy2IndexList, dbVPSNum, doVPSNum, dbEdgePSCounter, doEdgePSCounter, dWorkLoadPSCounter, workLoadNum, dbEdgeList, doEdgeList, &dSegmentIntersectJoinFlag);

    // PrintPairs(djxyVector, dPiPFlag, pairNum); // how to get pair IDs *************


    /*
    How to get the pairs in GPU
    djxyVector[]: ID pairs of the intersecting polygons from base and overlay {b_i,o_i}
    dPiPFlag: if a pair is intersecting, the flag vlues is 1. 
    pairNum: long value. Contains the total # intersecting pairs
    */

// GPUPrintVector(pairNum2, dEdgeIntersectCounter, 1);

    printf("\n\tActual number of intersected polygon pairs: %ld\n", pairNum3);
    float runningTime_GPU_CEI;
    Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_CEI);
    printf("\nGPU Running Time for Counting Edge Intersecions: %f (%f)\n", runningTime_GPU_CEI, Join_Total_Time_GPU);

// return;

    StartTimer(&start_GPU, &stop_GPU);


    float runningTime_GPU_EI;
    Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_EI);
    printf("\nGPU Running Time for Computing Edge Intersections: %f (%f)\n", runningTime_GPU_EI, Join_Total_Time_GPU);
//------------------------------------------------------------------------------

    // GPUPrintVector(eiNum, dSegmentIntersectJoinFlag, 1);

    // return;
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//cudaFree(djxyCounter);

//GCMF + GH merge sample code
// int baseID=260, overlayID=11589;
// double *polyPX, *polyPY, *polyQX, *polyQY;
// copyPolygon(&polyPX, &polyPY, &polyQX, &polyQY,
//             bVNum, bVPSNum, baseCoords, 
//             oVNum, oVPSNum, overlayCoords,
//             baseID, overlayID);

    // =========================================================================
    // Assign additional pointers to the input data arrays to be used in the GH code CPU
    *pBaseCoords=bCoords;
    *pOverlayCoords=oCoords;
    *pBVNum=dbVNum;
    *pBVPSNum=dbVPSNum;
    *pOVNum=doVNum; 
    *pOVPSNum=doVPSNum;
    // =========================================================================

*pIDList=(int *)malloc(sizeof(int)*pairNum);
*qIDList=(int *)malloc(sizeof(int)*pairNum);
CopyPairsToCPU(*pIDList, *qIDList, totalPairNum, djxyVector, dPiPFlag, pairNum); 

// cudaThreadExit(); ********** resets cuda environment
//==============================================================================
return 0;
}
