#include "GPU_Join.h"


//====================================== SortBaseMBROverlap ====================================================
long SortBaseMBROverlap(long bPolNum, long oPolNum, mbr_t* dbMBR, mbr_t* doMBR, int** djxyCounter, int** djxyVector, char dimSort, char dimSelect){
    cudaEvent_t start_GPU, stop_GPU;
    cudaError_t cudaMemError;
    cmbr_t* dxMBR, *dyMBR;
    int *dxMBRIndex, *dyMBRIndex, *dxSortIndex, *dySortIndex, *dxSortIndex2, *dySortIndex2, polNum=bPolNum+oPolNum;
    int *dxMBRLoadCounter, *dxMBRLoadPSCounter, *dyMBRLoadCounter, *dyMBRLoadPSCounter;

    //---------------------------------------------------- GPU Memory Assign ------------------------------------------------------------
    cudaMemError=cudaMalloc((void**)&dxMBR, 2*sizeof(cmbr_t)*polNum);
    cudaMemError=cudaMalloc((void**)&dxMBRIndex, 2*sizeof(int)*polNum);
    cudaMemError=cudaMalloc((void**)&dxSortIndex, 2*sizeof(int)*polNum);
    cudaMemError=cudaMalloc((void**)&dxSortIndex2, 2*sizeof(int)*polNum);
    cudaMemError=cudaMalloc((void**)&dxMBRLoadCounter, sizeof(int)*polNum);
    cudaMemError=cudaMalloc((void**)&dyMBR, 2*sizeof(cmbr_t)*polNum);
    cudaMemError=cudaMalloc((void**)&dyMBRIndex, 2*sizeof(int)*polNum);
    if(dimSort==2){
      cudaMemError=cudaMalloc((void**)&dyMBRLoadCounter, sizeof(int)*polNum);
      cudaMemError=cudaMalloc((void**)&dySortIndex, 2*sizeof(int)*polNum);
      cudaMemError=cudaMalloc((void**)&dySortIndex2, 2*sizeof(int)*polNum);
    }
   //------------------------------------------------------------------------------------------------------------------------------------


    StartTimer(&start_GPU, &stop_GPU);


   //------------------------------------------------------ Sort MBRs -------------------------------------------------------------------
    dim3 bDim_Init(256,1,1), gDim_Init(2*polNum/1000+1, 10, 1);    
    if(dimSort==2 || dimSelect==0)InitializeData<<<gDim_Init, bDim_Init>>>(bPolNum, oPolNum, dbMBR, doMBR, dxMBR, dyMBR); 
    else InitializeData<<<gDim_Init, bDim_Init>>>(bPolNum, oPolNum, dbMBR, doMBR, dyMBR, dxMBR); 
    GPUSync("InitializeData");
    RadixSort(dxMBR, dyMBR, dxMBRIndex, dyMBRIndex, dxSortIndex, dySortIndex, dxSortIndex2, dySortIndex2, 0, MAX_DIGITS-1, MAX_DIGITS, 2*(polNum), dimSort);

    //GPUPrintVector(2*polNum, dyMBR, 1);
    //GPUPrintSortedData(2*polNum, dxMBR, dxSortIndex);
    //GPUIndexCoverageTest(2*polNum, dxSortIndex);
    //PrintGPUVector(2*polNum, dxMBR, 1);
    //GPUMBRIndexTest(2*polNum, dxMBRIndex, dxSortIndex, dxMBR);
   //-----------------------------------------------------------------------------

/*
   //-------------------------------- Choose Lighter Dimension and Calculate Workload per Base MBR --------------------------------------
    dim3 bDim_MBRLoad(1024,1,1), gDim_MBRLoad(polNum/1000+1, 1, 1);    
    CalculateMBRLoad<<<gDim_MBRLoad, bDim_MBRLoad>>>(dxMBRIndex, dyMBRIndex, dxMBRLoadCounter, dyMBRLoadCounter, (polNum), dimSort);
    PrefixSum(polNum, dxMBRLoadCounter, dyMBRLoadCounter, &dxMBRLoadPSCounter, &dyMBRLoadPSCounter, 1, dimSort);

    int xWorkLoad, yWorkLoad, minWorkLoad, chosenDim=0;
    cudaMemcpy(&xWorkLoad,dxMBRLoadPSCounter+(polNum-1), sizeof(int), cudaMemcpyDeviceToHost);
    minWorkLoad=xWorkLoad;
    if(dimSort==2){
       cudaMemcpy(&yWorkLoad,dyMBRLoadPSCounter+(polNum-1), sizeof(int), cudaMemcpyDeviceToHost);
       if(minWorkLoad>yWorkLoad){minWorkLoad=yWorkLoad;chosenDim=1;}
    }
    else chosenDim=dimSelect;
    cudaFree(dxMBRLoadCounter);
    cudaFree(dxMBRLoadPSCounter);
    if(dimSort==2){
      cudaFree(dyMBRLoadPSCounter);
      cudaFree(dyMBRLoadCounter);
    }
    int jvSize=2*(minWorkLoad+(polNum)*INTERSECT_CONST);
    //RunTestModule();
   //-------------------------------------------------------------------------------------------------------------------------------------
*/
int chosenDim=0;


    float runningTime_GPU_build;
    StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_build);
    printf("\n\tSorting %dD: %f \n",dimSort, runningTime_GPU_build);

   //----------------------------------------------------------- Calculate MBR Count -----------------------------------------------------
    StartTimer(&start_GPU, &stop_GPU);

    long *djxyPSCounter;
    cudaMemError=cudaMalloc((void**) djxyCounter, sizeof(int)*polNum);
    GPUMAllocCheck(cudaMemError, "djxyCounter");

    dim3 bDim_CountMBR(1024,1,1), gDim_CountMBR((polNum)/1000+1, 1000, 1);    
    //Even digit case so SortIndex is used.
    if(dimSort==1 || chosenDim==0)
       CountSortBaseMBROverlapLoad<<<gDim_CountMBR, bDim_CountMBR>>>(dxMBR, dyMBR, bPolNum, oPolNum, dxSortIndex, dySortIndex, dxMBRIndex, dyMBRIndex, *djxyCounter, dimSort); 
    else
       CountSortBaseMBROverlapLoad<<<gDim_CountMBR, bDim_CountMBR>>>(dyMBR, dxMBR, bPolNum, oPolNum, dySortIndex, dxSortIndex, dyMBRIndex, dxMBRIndex, *djxyCounter, dimSort); 
    GPUSync("ERROR (CountMBROverlapLoad):");

    PrefixSum(polNum, *djxyCounter, NULL, &djxyPSCounter, NULL, 1, 1);

    long *pairNum;
    CopyFromGPU((void**)&pairNum, (djxyPSCounter)+polNum-1, sizeof(long), 1);

    //GPUPrefixsumTest(*djxyCounter, djxyPSCounter, polNum, 0);
    //GPUPrintVector(polNum, *djxyCounter, 1);
    //printf("\nPairNum: %ld \n", *pairNum);
    //return 0;
    float runningTime_GPU_overlap_Cnt;
    StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_overlap_Cnt);
    printf("\n\tCounting MBR intersection : %f \n", runningTime_GPU_overlap_Cnt);
   //-------------------------------------------------------------------------------------------------------------------------------------


   //------------------------------------------------------- MBRs Overlap ---------------------------------------------------------------

    cudaMemError=cudaMalloc((void**)djxyVector, 2*sizeof(int)**pairNum);    
    GPUMAllocCheck(cudaMemError, "djxyVector");
    //Small:128		Medium:512		Large:
    int thPerBlock;
    switch(DATASET){
      case 1:
        thPerBlock=128;
        break;
      case 2:
        thPerBlock=512;
        break;
      case 3:
        thPerBlock=128;
        break;
      default:
        thPerBlock=128;
        break;
    }
    dim3 bDim_GetOLMBR(thPerBlock,1,1), gDim_GetOLMBR((polNum)/1000+1, 1000, 1);    

    //Even digit case so SortIndex is used.
    if(dimSort==1 || chosenDim==0)
       SortBaseMBROverlapLoadCalculated<<<gDim_GetOLMBR, bDim_GetOLMBR>>>(dxMBR, dyMBR, bPolNum, oPolNum, dxSortIndex, dySortIndex, dxMBRIndex, dyMBRIndex, *djxyCounter, *djxyVector, djxyPSCounter, dimSort); 
    else
       SortBaseMBROverlapLoadCalculated<<<gDim_GetOLMBR, bDim_GetOLMBR>>>(dyMBR, dxMBR, bPolNum, oPolNum, dySortIndex, dxSortIndex, dyMBRIndex, dxMBRIndex, *djxyCounter, *djxyVector, djxyPSCounter, dimSort); 
    GPUSync("ERROR (SortBaseMBROverlapLoadCalculated):");

    //GPUPrintVector(2**pairNum, *djxyVector, 0);
    //CountPairs(jxyVector, jxyCounter, polNum, GPU_MAX_JOIN_PER_DIM);
    //PrintBucketPairVector(jxyVector, jxyCounter, polNum, GPU_MAX_JOIN_PER_DIM, 0);*/
    float runningTime_GPU_overlap2;
    StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_overlap2);
    printf("\n\tComputing MBR intersection (new approach %dD [dim:%c] ): %f \n",dimSort, 'X'+chosenDim, runningTime_GPU_overlap2);
    //-------------------------------------------------------------------------------------------------------------------------------------

    cudaFree(dxMBRIndex);
    cudaFree(dyMBRIndex);
    cudaFree(dxSortIndex);
    cudaFree(dxSortIndex2);
    cudaFree(djxyPSCounter);
    if(dimSort==2){
      cudaFree(dySortIndex);
      cudaFree(dySortIndex2);
    }
    return(*pairNum);
}
//=======================================================================================

//=========================== CountCMF =============================
void CountCMF(coord_t *bCoords, coord_t * oCoords, long pairNum, int* jxyVector, int *joinFlag, int* bVNum, int* oVNum, long *bVPSNum, long* oVPSNum, coord_t* cMBR, long** bEdgePSCounter, long** oEdgePSCounter, long** workLoadPSCounter, int** jxy2IndexList, poly_size_t** bEdgeList, poly_size_t** oEdgeList, long* eiNum, long* workLoadNum){
    long *joinPSFlag, *eiNum0;
    int *jTempIndexList, *bEdgeCounter, *oEdgeCounter; 
    
    PrefixSum(pairNum, joinFlag, NULL, &joinPSFlag, NULL, 1, 1);
    CopyFromGPU((void**)&eiNum0, joinPSFlag+pairNum-1, sizeof(long), 1);
    cudaError_t cudaMemError=cudaMalloc((void**)&jTempIndexList, sizeof(int)**eiNum0);
    GPUMAllocCheck(cudaMemError, "jTempIndexList");
    printf("\n\tNumber of candidates for CMF filter: %ld\n", *eiNum0);

    int gD=sqrt(pairNum/1000)+1;
    dim3 gDim_T(gD ,gD ,1);        
    dim3 bDim_T(1024,1,1);    
    Make_Filtered_Index_List<<<gDim_T, bDim_T>>>(pairNum, joinPSFlag, NULL, jTempIndexList);
    GPUSync("ERROR (Make_Filtered_Index_List):");


    cudaMemError=cudaMalloc((void**)&bEdgeCounter, sizeof(int)*pairNum);
    GPUMAllocCheck(cudaMemError, "bEdgeCounter");
    cudaMemError=cudaMalloc((void**)&oEdgeCounter, sizeof(int)*pairNum);
    GPUMAllocCheck(cudaMemError, "oEdgeCounter");
    //cudaMemError=cudaMalloc((void**)workLoadPSCounter, sizeof(long)*pairNum);
    //GPUMAllocCheck(cudaMemError, "workLoadPSCounter");
   
    InitializeVector2(pairNum, bEdgeCounter, oEdgeCounter, 0);

    //Small:128		Medium:64	Large:
    int thPerBlock;
    switch(DATASET){
      case 1:
        thPerBlock=128;
        break;
      case 2:
        thPerBlock=64;
        break;
      case 3:
        thPerBlock=128;
        break;
      default:
        thPerBlock=128;
        break;
    }
    gD=sqrt(*eiNum0)+1;
    dim3 bDim_CMF(thPerBlock,2,1);
    dim3 gDim_CMF(gD ,gD ,1);        
    //gD=sqrt(2**eiNum0)+1;
    //dim3 bDim_CMF(128,1,1);
    //Count_CMF_1D<<<gDim_CMF, bDim_CMF>>>(bCoords, oCoords, cMBR, *eiNum0, jxyVector, jTempIndexList, bVNum, oVNum, bVPSNum, oVPSNum, bEdgeCounter, oEdgeCounter, joinFlag);
    Count_CMF<<<gDim_CMF, bDim_CMF>>>(bCoords, oCoords, cMBR, *eiNum0, jxyVector, jTempIndexList, bVNum, oVNum, bVPSNum, oVPSNum, bEdgeCounter, oEdgeCounter, joinFlag);
    GPUSync("ERROR (Count_CMF):");

    PrefixSum(pairNum, bEdgeCounter, oEdgeCounter, bEdgePSCounter, oEdgePSCounter, 1, 2);
    GPUSync("ERROR (EdgePSCounter):");

    long *bEdgeNum, *oEdgeNum;
    CopyFromGPU((void**)&bEdgeNum, (*bEdgePSCounter)+(pairNum)-1, sizeof(long), 1);
    CopyFromGPU((void**)&oEdgeNum, (*oEdgePSCounter)+(pairNum)-1, sizeof(long), 1);
    printf("\n\tTotal number of base edges: %ld \t Total number of overlay edges: %ld\n", *bEdgeNum, *oEdgeNum);
    cudaMemError=cudaMalloc((void**)bEdgeList, sizeof(poly_size_t)**bEdgeNum);
    GPUMAllocCheck(cudaMemError, "dbEdgeList");
    cudaMemError=cudaMalloc((void**)oEdgeList, sizeof(poly_size_t)**oEdgeNum);
    GPUMAllocCheck(cudaMemError, "doEdgeList");

    cudaFree(joinPSFlag);

//---------------------------- Calculating the workload -------------------------------
    int *workLoadCounter;
    cudaMemError=cudaMalloc((void**)&workLoadCounter, sizeof(int)*pairNum);
    GPUMAllocCheck(cudaMemError, "workLoadCounter");
    DotProduct(pairNum, bEdgeCounter, oEdgeCounter, workLoadCounter, 1);   

// ------------------------------------------------------------------------------------

    PrefixSum(pairNum, joinFlag, workLoadCounter, &joinPSFlag, workLoadPSCounter, 1, 2);
    CopyFromGPU((void**)&eiNum, joinPSFlag+pairNum-1, sizeof(long), 0);
    CopyFromGPU((void**)&workLoadNum, (*workLoadPSCounter)+pairNum-1, sizeof(long), 0);
    printf("\n\tNumber of candidates after CMF filter: %ld\n\tTotal Number of Edge Pairs: %ld\n", *eiNum, *workLoadNum);


    cudaMemError=cudaMalloc((void**)jxy2IndexList, sizeof(int)**eiNum);
    GPUMAllocCheck(cudaMemError, "jxy2IndexList");

    gD=sqrt(pairNum/1000)+1;
    dim3 gDim_CV(gD ,gD ,1);        
    dim3 bDim_CV(1024,1,1);    
    Make_Filtered_Index_List<<<gDim_CV, bDim_CV>>>(pairNum, joinPSFlag, NULL, *jxy2IndexList);
    GPUSync("ERROR (Make_Filtered_Index_List):");

//printf("\nPN: %ld\n", *pairNum2);
//GPUPrintVector(pairNum, *jxyFlag ,0);
//GPUPrintVector(*eiNum, *jxy2IndexList ,0);
//PrintPairs(jxyVector, *pipFlag, pairNum);
//exit(0);
    cudaFree(joinPSFlag);
    cudaFree(jTempIndexList);
    return;

}
//======================================================================================
//=========================== Apply_CMF =============================
void ApplyCMF(coord_t *bCoords, coord_t * oCoords, long pairNum, int* jxyVector, long eiNum, int * jxy2IndexList, int* bVNum, int* oVNum, long *bPFVNum, long* oPFVNum, coord_t* cMBR, long* bEdgePSCounter, long* oEdgePSCounter, poly_size_t *bEdgeList, poly_size_t *oEdgeList){

    int gD_ACMF=sqrt(pairNum)+1;
    //Small:128	Medium:64	Large:
    int thPerBlock;
    switch(DATASET){
      case 1:
        thPerBlock=128;
        break;
      case 2:
        thPerBlock=64;
        break;
      case 3:
        thPerBlock=512;
        break;
      default:
        thPerBlock=512;
        break;
    }

    dim3 bDim_ACMF(thPerBlock,2,1);    
    dim3 gDim_ACMF(gD_ACMF ,gD_ACMF ,1);        
    Apply_CMF<<<gDim_ACMF, bDim_ACMF>>>(bCoords, oCoords, eiNum, jxyVector, jxy2IndexList, bVNum, oVNum, bPFVNum, oPFVNum, cMBR, bEdgePSCounter, oEdgePSCounter, bEdgeList, oEdgeList);
    GPUSync("ERROR (Apply_CMF):");


//GPUPrintVector(4297, bEdgeList ,0);
//GPUPrintVector(4510, oEdgeList ,0);
//exit(0);
    return;
}
//=======================================================================================


//=========================== CountEdgeIntersect =============================
void CountEdgeIntersect(coord_t *bCoords, coord_t * oCoords, long pairNum, int* jCompactVector, int *jxy2IndexList, int* bVNum, int* oVNum, long *bPFVNum, long* oPFVNum, long* bEdgePSCounter, long* oEdgePSCounter, poly_size_t *bEdgeList, poly_size_t* oEdgeList, int** edgeIntersectCounter, int** jxy3IndexList, long *pairNum3){
    int*jxy3Flag;
    long *jxy3PSFlag;
    cudaError_t cudaMemError=cudaMalloc((void**)&jxy3Flag, sizeof(int)*pairNum);

    cudaMemError=cudaMalloc((void**)edgeIntersectCounter, sizeof(int)*pairNum);
    GPUMAllocCheck(cudaMemError, "edgeIntersectCounter");

    int gD_CEI=sqrt(pairNum)+1;
    dim3 bDim_CEI(1024,1,1);    
    dim3 gDim_CEI(gD_CEI ,gD_CEI ,1);        
    Count_EdgeIntersect<<<gDim_CEI, bDim_CEI>>>(bCoords, oCoords, pairNum, jCompactVector, jxy2IndexList, bPFVNum, oPFVNum, bEdgePSCounter, oEdgePSCounter, bEdgeList, oEdgeList, *edgeIntersectCounter, jxy3Flag);
    GPUSync("ERROR (Count_EdgeIntersect):");

    PrefixSum(pairNum, jxy3Flag, NULL, &jxy3PSFlag, NULL, 1, 1);
    CopyFromGPU((void**)&pairNum3, jxy3PSFlag+pairNum-1, sizeof(long), 0);

    cudaMemError=cudaMalloc((void**)jxy3IndexList, sizeof(int)**pairNum3);
    GPUMAllocCheck(cudaMemError, "jxy3IndexList");

    int gD_CV=sqrt(pairNum/1000)+1;
    dim3 gDim_CV(gD_CV ,gD_CV ,1);        
    dim3 bDim_CV(1024,1,1);    
    Make_Filtered_Index_List<<<gDim_CV, bDim_CV>>>(pairNum, jxy3PSFlag, jxy2IndexList, *jxy3IndexList);
    GPUSync("ERROR (Make_Filtered_Index_List):");

    //PUPrintVector(*pairNum3, *jxy3IndexList, 0);

    cudaFree(jxy3Flag);
    cudaFree(jxy3PSFlag);

    return; 
}
//=======================================================================================


//=========================== SegmentIntersectJoin2 =============================
long SegmentIntersectJoin2(coord_t *bCoords, coord_t * oCoords, long pairNum, int* jCompactVector, int *jxy2IndexList, long *bPFVNum, long* oPFVNum, long* bEdgePSCounter, long* oEdgePSCounter, long *workLoadPSCounter, long workLoadNum, poly_size_t *bEdgeList, poly_size_t* oEdgeList, int** segIntersectJoinFlag){

    long*segJoinPSFlag;
    cudaError_t cudaMemError=cudaMalloc((void**)segIntersectJoinFlag, sizeof(int)*pairNum);

    InitializeVector(pairNum, *segIntersectJoinFlag, 0);

    int gD_CEI=sqrt(workLoadNum/(EDGE_PER_THREAD*512))+1;
    dim3 bDim_CEI(512,1,1);    
    dim3 gDim_CEI(gD_CEI ,gD_CEI ,1);        
    Segment_Intersect_Join2<<<gDim_CEI, bDim_CEI>>>(bCoords, oCoords, pairNum, jCompactVector, jxy2IndexList, bPFVNum, oPFVNum, bEdgePSCounter, oEdgePSCounter, workLoadPSCounter, workLoadNum, bEdgeList, oEdgeList, *segIntersectJoinFlag);
    GPUSync("ERROR (Segment_Intersect_Join):");
    long *joinPairNum;
    PrefixSum(pairNum, *segIntersectJoinFlag, NULL, &segJoinPSFlag, NULL, 1, 1);
    CopyFromGPU((void**)&joinPairNum, segJoinPSFlag+pairNum-1, sizeof(long), 1);

    //GPUPrintVector(pairNum, *segIntersectJoinFlag, 1);
    //exit(0);

    cudaFree(segJoinPSFlag);

    return *joinPairNum; 
}
//=======================================================================================


//=========================== SegmentIntersectJoin =============================
long SegmentIntersectJoin(coord_t *bCoords, coord_t * oCoords, long pairNum, int* jCompactVector, int *jxy2IndexList, long *bPFVNum, long* oPFVNum, long* bEdgePSCounter, long* oEdgePSCounter, poly_size_t *bEdgeList, poly_size_t* oEdgeList, int** segIntersectJoinFlag){
    long*segJoinPSFlag;
    cudaError_t cudaMemError=cudaMalloc((void**)segIntersectJoinFlag, sizeof(int)*pairNum);

    int gD_CEI=sqrt(pairNum)+1;
    //Small:		Medium:		Large:
    int thPerBlock;
    switch(DATASET){
      case 1:
        thPerBlock=64;
        break;
      case 2:
        thPerBlock=64;
        break;
      case 3:
        thPerBlock=512;
        break;
      default:
        thPerBlock=512;
        break;
    }

    dim3 bDim_CEI(thPerBlock,1,1);    
    dim3 gDim_CEI(gD_CEI ,gD_CEI ,1);        
    Segment_Intersect_Join<<<gDim_CEI, bDim_CEI>>>(bCoords, oCoords, pairNum, jCompactVector, jxy2IndexList, bPFVNum, oPFVNum, bEdgePSCounter, oEdgePSCounter, bEdgeList, oEdgeList, *segIntersectJoinFlag);
    GPUSync("ERROR (Segment_Intersect_Join):");

    long *joinPairNum;
    PrefixSum(pairNum, *segIntersectJoinFlag, NULL, &segJoinPSFlag, NULL, 1, 1);
    CopyFromGPU((void**)&joinPairNum, segJoinPSFlag+pairNum-1, sizeof(long), 1);

    //GPUPrintVector(pairNum, *segIntersectJoinFlag, 1);
    //exit(0);

    cudaFree(segJoinPSFlag);

    return *joinPairNum; 
}
//=======================================================================================

//=========================== GetCMF =============================
void GetCMBR(long pairNum, int* jxyVector, coord_t* bMBR, coord_t* oMBR, coord_t** cMBR, int** jpipIndexList, int** pipFlag, char**pipType, int** joinFlag, long* pipNum){
    long *pipPSFlag;

    cudaError_t cudaMemError=cudaMalloc((void**)joinFlag, sizeof(int)*pairNum);
    GPUMAllocCheck(cudaMemError, "joinFlag");
    cudaMemError=cudaMalloc((void**)pipFlag, sizeof(int)*pairNum);
    GPUMAllocCheck(cudaMemError, "pipFlag");
    cudaMemError=cudaMalloc((void**)pipType, sizeof(char)*pairNum);
    GPUMAllocCheck(cudaMemError, "pipType");
    cudaMemError=cudaMalloc((void**)cMBR, 4*sizeof(coord_t)*pairNum);    
    GPUMAllocCheck(cudaMemError, "dcMBR");


    int gD_CCMF=sqrt(pairNum/1000)+1;
    dim3 bDim_CCMF(1024,1,1);    
    dim3 gDim_CCMF(gD_CCMF ,gD_CCMF ,1);        
    Get_CMBR<<<gDim_CCMF, bDim_CCMF>>>(pairNum, jxyVector, *pipFlag, *pipType, bMBR, oMBR, *cMBR, *joinFlag);
    GPUSync("ERROR (Get_CMBR):");

    PrefixSum(pairNum, *pipFlag, NULL, &pipPSFlag, NULL, 1, 1);
    CopyFromGPU((void**)&pipNum, pipPSFlag+pairNum-1, sizeof(long), 0);
    printf("\n\tNumber of Point in Polygon candidates: %ld\n", *pipNum);

    cudaMemError=cudaMalloc((void**)jpipIndexList, sizeof(int)**pipNum);
    GPUMAllocCheck(cudaMemError, "jpipIndexList");


    
    int gD_CV=sqrt(pairNum/1000)+1;
    dim3 gDim_CV(gD_CV ,gD_CV ,1);        
    dim3 bDim_CV(1024,1,1);    
    Make_Filtered_Index_List<<<gDim_CV, bDim_CV>>>(pairNum, pipPSFlag, NULL, *jpipIndexList);
    GPUSync("ERROR (Make_Filtered_Index_List):");

    return;
}
//=======================================================================================


//============================ PointInPolygonTest ==============================
long PointInPolygonTest(coord_t *bCoords, coord_t* oCoords, long pairNum, long pipNum, int* jxyVector, int* jPiPIndexList, char* pipType, long* bVPSNum, long* oVPSNum, int* pipFlag, int* joinFlag){

    int gD_PiP=sqrt(pipNum)+1;
    //Small:256		Medium:128		Large:
    int thPerBlock;
    switch(DATASET){
      case 1:
        thPerBlock=256;
        break;
      case 2:
        thPerBlock=128;
        break;
      case 3:
        thPerBlock=512;
        break;
      default:
        thPerBlock=512;
        break;
    }
    //dim3 bDim_PiP(thPerBlock,1,1);    
    dim3 bDim_PiP(thPerBlock,1,1);    
    dim3 gDim_PiP(gD_PiP ,gD_PiP ,1);        
    Point_In_Polygon_Test<<<gDim_PiP, bDim_PiP>>>(bCoords, oCoords, pipNum, jxyVector, jPiPIndexList, pipType, bVPSNum, oVPSNum, pipFlag, joinFlag);
    GPUSync("ERROR (Point_In_Polygon_Test):");

    long *jNum, *pipPSFlag;
    PrefixSum(pairNum, pipFlag, NULL, &pipPSFlag, NULL, 1, 1);
    CopyFromGPU((void**)&jNum, pipPSFlag+pairNum-1, sizeof(long), 1);
    //GPUPrintVector(pairNum, *segIntersectJoinFlag, 1);
    return(*jNum);
}
