
#include "stdio.h"
#include "GPU_Manage.h"
#include "Types.h"
#include "Constants.h"
#include "GPU_Test.h"
#include "GPU_MBR.h"
#include "IO.h"
#include "SEQ_Overlay.h"
#include "GPU_Utility.h"
#include "Data_Visualization.h"
#include "Join.h"

// int main() {
//   return 0;
// }

int ST_Intersect(long bPolNum, long oPolNum, coord_t* baseCoords, coord_t* overlayCoords, int* bVNum, int* oVNum, long *bVPSNum, long *oVPSNum, 
                 mbr_t* seqMBR, mbr_t* seqOMBR, coord_t *seqMBR2, coord_t* seqOMBR2, int * jPairs){

    printf("In ST_Intersect\n");
    fflush(stdout);
    
    long bVNumSum=bVPSNum[bPolNum-1], oVNumSum=oVPSNum[oPolNum-1];
    float Join_Total_Time_SEQ=0, Join_Total_Time_GPU=0;
    cudaError_t cudaMemError;
    int dimSort=1, dimSelect=0, retVal=0;

//=============================== SEQUENTIAL RUN ===============================
//------------------------------------------------------------------------------    

//PrintPolygon(baseCoords+2*bVPSNum[1485], bVNum[1486]);
//printf("\n\n\n");
//PrintPolygon(overlayCoords+2*oVPSNum[10], oVNum[11]);
//return;

//=========================== Reseting GPU device ==============================
    //====================== Transfering data to GPU ==========================
    cudaStream_t gStream;
    cudaError_t stEr=cudaStreamCreate(&gStream);
    if(stEr!=cudaSuccess){printf("\nError in creating stream!\n");return(0);}
    cudaEvent_t start_GPU, stop_GPU;
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

    if(DEBUG_MODE){
      float runningTime_GPU_TransferData;
      Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_TransferData);
      printf("\n\nGPU running time for transfering data to GPU: %f (%f)\n",runningTime_GPU_TransferData, Join_Total_Time_GPU);
    }
//==============================================================================


//--------------------------- Find Overlaping MBRs (novel approach) ---------------------------
    StartTimer(&start_GPU, &stop_GPU);

    int *djxyCounter, *djxyVector, polNum=bPolNum+oPolNum; 
    cudaMemError=cudaMalloc((void**)&djxyCounter,sizeof(int)*(polNum));

    long pairNum=SortBaseMBROverlap(gStream, bPolNum, oPolNum, dbMBR, doMBR, &djxyCounter, &djxyVector, dimSort, dimSelect);
    float runningTime_GPU_overlap2;
   
    if(DEBUG_MODE){
      printf("\n\n\tPolygon pairs candidate: %ld\n", pairNum);
      Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_overlap2);
      printf("\nGPU Running Time For Computing MBR intersection (new approach %dD [dim:%c] ): %f (%f)\n",dimSort, 'X', runningTime_GPU_overlap2, Join_Total_Time_GPU);
    }
    cudaFree(doMBR);
    cudaFree(dbMBR);
    cudaFree(djxyCounter);
    if(pairNum==0)return(0);
//------------------------------------------------------------------------------

//--------------------------- CMF filter for Polygon Test operation --------------------------
    StartTimer(&start_GPU, &stop_GPU);
    int *djxy2IndexList, *djPiPIndexList;
    char* dPiPType, *dPiPFlag, *djoinFlag;
    long eiNum, pairNum3, /* *pipNum */ pipNum, workLoadNum;
    coord_t *dcMBR, *dbMBR2, *doMBR2;
    CopyToGPU((void**)&doMBR2, seqOMBR2, sizeof(coord_t)*oPolNum*4, "doMBR2", 1);
    CopyToGPU((void**)&dbMBR2, seqMBR2, sizeof(coord_t)*bPolNum*4, "dbMBR2", 1);
    GPUSync("Transfering data to GPU");

    GetCMBR(gStream, pairNum, djxyVector, dbMBR2, doMBR2, &dcMBR, &djPiPIndexList, &dPiPFlag, &dPiPType, &djoinFlag, &pipNum);
 
    if(DEBUG_MODE){
       float runningTime_GPU_PiPCMF;
       Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_PiPCMF);
       printf("\nGPU Running Time for CMF Filter for Point in Polygon Test:  %f (%f)\n", runningTime_GPU_PiPCMF, Join_Total_Time_GPU);
    }
//------------------------------------------------------------------------------
//--------------------------- Point in Polygon Test operation --------------------------
    StartTimer(&start_GPU, &stop_GPU);
    long wNum;
    
    // wNum=PointInPolygonTest(bCoords, oCoords, pairNum, *pipNum, djxyVector, djPiPIndexList, dPiPType, dbVPSNum, doVPSNum, dPiPFlag, djoinFlag);
    wNum=PointInPolygonTest(bCoords, oCoords, pairNum, pipNum, djxyVector, djPiPIndexList, dPiPType, dbVPSNum, doVPSNum, dPiPFlag, djoinFlag);

    if(DEBUG_MODE)printf("\n\tNumber of within pairs: %ld\n", wNum);
    retVal+=wNum;

    //PrintPairs(djxyVector, dPiPFlag, pairNum);
//GPUPrintVector(pairNum2, dEdgeIntersectCounter, 1);

    if(DEBUG_MODE){
      float runningTime_GPU_PiP;
      Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_PiP);
      printf("\nGPU Running Time for Point in Polygon Test: %f (%f)\n", runningTime_GPU_PiP, Join_Total_Time_GPU);
    }
 
//------------------------------------------------------------------------------


//--------------------------- Applying Common MBR Filtering (novel approach) ---------------------------
    StartTimer(&start_GPU, &stop_GPU);
    poly_size_t *dbEdgeList, *doEdgeList;
    long *dbEdgePSCounter, *doEdgePSCounter, *dWorkLoadPSCounter;


     CountCMF(gStream, bCoords, oCoords, pairNum, djxyVector, djoinFlag, dbVNum, doVNum, dbVPSNum, doVPSNum, dcMBR, &dbEdgePSCounter, &doEdgePSCounter, &dWorkLoadPSCounter, &djxy2IndexList, &dbEdgeList, &doEdgeList, &eiNum, &workLoadNum);

    //printf("\n\tPolygon pair candidates after Applying CMF filter: %ld\n", eiNum);
    if(DEBUG_MODE){
      float runningTime_GPU_CCMF;
      Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_CCMF);
      printf("\nGPU Running Time for Counting Common MBR Filter: %f (%f)\n", runningTime_GPU_CCMF, Join_Total_Time_GPU);
    }


    StartTimer(&start_GPU, &stop_GPU);

    ApplyCMF(bCoords, oCoords, pairNum, djxyVector, eiNum, djxy2IndexList, dbVNum, doVNum, dbVPSNum, doVPSNum, dcMBR, dbEdgePSCounter, doEdgePSCounter, dbEdgeList, doEdgeList);
    //GPUPrintVector(pairNum*2, djxyVector, 0);
    //GPUPrefixsumTest(dbEdgeCounter, dbEdgePSCounter, pairNum, 1);
    //GPUPrefixsumTest(doEdgeCounter, doEdgePSCounter, pairNum, 1);

    if(DEBUG_MODE){
      float runningTime_GPU_ACMF;
      Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_ACMF);
      printf("\nGPU Running Time for Applying Common MBR Filter : %f (%f)\n", runningTime_GPU_ACMF, Join_Total_Time_GPU);
    }
    cudaFree(dcMBR);

//------------------------------------------------------------------------------

//--------------------------- Join/Overlay operations --------------------------
    StartTimer(&start_GPU, &stop_GPU);
    char* dSegmentIntersectJoinFlag;
     pairNum3=SegmentIntersectJoin(bCoords, oCoords, eiNum, djxyVector, djxy2IndexList, dbVPSNum, doVPSNum, dbEdgePSCounter, doEdgePSCounter, dbEdgeList, doEdgeList, &dSegmentIntersectJoinFlag);
    //pairNum3=SegmentIntersectJoin2(bCoords, oCoords, eiNum, djxyVector, djxy2IndexList, dbVPSNum, doVPSNum, dbEdgePSCounter, doEdgePSCounter, dWorkLoadPSCounter, workLoadNum, dbEdgeList, doEdgeList, &dSegmentIntersectJoinFlag);
    //PrintPairs(djxyVector, dPiPFlag, pairNum);

    retVal+=pairNum3;

    if(DEBUG_MODE){
      printf("\n\tActual number of intersected polygon pairs: %ld\n", pairNum3);
      float runningTime_GPU_CEI;
      Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_CEI);
      printf("\nGPU Running Time for Counting Edge Intersecions: %f (%f)\n", runningTime_GPU_CEI, Join_Total_Time_GPU);
    }
//------------------------------------------------------------------------------

    GPUSync("SegmentIntersect");


   char *pipFlag, *joinFlag, *jxyVector, *jxyIndexList, mappedIndx; 
   CopyFromGPU((void**)&pipFlag, dPiPFlag, sizeof(char)*pairNum, 1);
   CopyFromGPU((void**)&joinFlag, dSegmentIntersectJoinFlag, sizeof(char)*eiNum, 1);
   CopyFromGPU((void**)&jxyIndexList, djxy2IndexList, sizeof(int)*eiNum, 1);
   CopyFromGPU((void**)&jxyVector, djxyVector, 2*sizeof(int)*pairNum, 1);
   jPairs=(int*)malloc(sizeof(int)*2*retVal);

   int indx=0;
   for(int i=0;i<pairNum;i++){
      if(pipFlag[i]==1){
         jPairs[indx*2]=jxyVector[2*i];
         jPairs[indx*2+1]=jxyVector[2*i+1];
         indx++;
     }
      else if(i<eiNum){
         if(joinFlag[i]==1){
            mappedIndx=jxyIndexList[i];
            jPairs[indx*2]=jxyVector[2*mappedIndx];
            jPairs[indx*2+1]=jxyVector[2*mappedIndx+1];
            indx++;
         }
      }
      if(indx>retVal){
        printf("\nToo many outputs!\n");
        break;
      }
   }

   cudaFree(dPiPFlag);
   cudaFree(djxyVector);
   cudaFree(djoinFlag);
   free(pipFlag);
   free(joinFlag);
   free(jxyVector);

   cudaThreadExit();
   return(retVal);

//==============================================================================
}
