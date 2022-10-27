#include "GPU_PrefixSum.h"
#include "GPU_Vector.h"
#include "GPU_RadixSort.h"

//=============================== DotProduct ================================
void Multiply(long elementNum, int* vector, int factor, int* result){
    int dpDim=sqrt(elementNum)+1;
    dim3 bDim_DP(1024, 1, 1), gDim_DP(dpDim, dpDim, 1);    
    //GPUMultiply<<<gDim_DP, bDim_DP>>>(elementNum, vector, factor, result);
    GPUSync("ERROR (Multiply)");
    return;
}
//==============================================================================


//=============================== DotProduct ================================
void DotProduct(long elementNum, int* vector1, char* vector2, int* result, char synced){
    int dpDim=sqrt(elementNum)+1;
    dim3 bDim_DP(1024, 1, 1), gDim_DP(dpDim, dpDim, 1);    
    GPUDotProduct<<<gDim_DP, bDim_DP>>>(elementNum, vector1, vector2, result);
    if(synced)GPUSync("ERROR (DotProduct)");
    return;
}
//==============================================================================



//=============================== DotProduct ================================
void DotProduct(long elementNum, int* vector1, int* vector2, int* result, char synced){
    int dpDim=sqrt(elementNum)+1;
    dim3 bDim_DP(1024, 1, 1), gDim_DP(dpDim, dpDim, 1);    
    GPUDotProduct<<<gDim_DP, bDim_DP>>>(elementNum, vector1, vector2, result);
    if(synced)GPUSync("ERROR (DotProduct)");
    return;
}
//==============================================================================


//============================== InitializeVector ==============================
void InitializeVector(long elementNum, int* vector1, int value){
    int dpDim=sqrt(elementNum/1024+1)+1;
    dim3 bDim_DP(1024, 1, 1), gDim_DP(dpDim, dpDim, 1);    
    Initialize_Vector<<<gDim_DP, bDim_DP>>>(elementNum, vector1, value);
    GPUSync("ERROR (Initialize_Vector)");
    return;
}
//==============================================================================


//============================= InitializeVector2 ==============================
void InitializeVector2(long elementNum, int* vector1, int* vector2, int value){
    int dpDim=sqrt(elementNum/1024+1)+1;
    dim3 bDim_DP(1024, 1, 1), gDim_DP(dpDim, dpDim, 1);    
    Initialize_Vector2<<<gDim_DP, bDim_DP>>>(elementNum, vector1, vector2, value);
    GPUSync("ERROR (Initialize_Vector)");
    return;
}
//==============================================================================


//================================== PrefixSum ==================================
void PrefixSum(long eNum, char* xVector, char* yVector, long** xPSVector, long** yPSVector, char isNew, char psDim){
    long* yPSVect;
    if(isNew){
      cudaError_t cudaMemError=cudaMalloc((void**)xPSVector, sizeof(long)*eNum);
      GPUMAllocCheck(cudaMemError, "xPSVector");
      if(psDim==2){
         cudaMemError=cudaMalloc((void**)yPSVector, sizeof(long)*eNum);
         GPUMAllocCheck(cudaMemError, "yPSVector");
      }
       else yPSVect=NULL;
 
     }
     else yPSVect=*yPSVector;
    int blk=sqrt(eNum/1000)+1, blk2=sqrt(eNum/1000000)+1, blk3=sqrt(eNum/1000000000)+1, blk4=sqrt(eNum/1000000000000)+1;
    dim3 bDim_PS(1024, 1, 1), gDim_PS(blk, blk, psDim); 
    dim3 gDim_PS2(blk2, blk2, psDim), gDim_PS3(blk3, blk3, psDim), gDim_PS4(blk4, blk4, psDim);    

    ComputeNewPrefixSum<<<gDim_PS, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 0, 1);
    if(eNum>1024)ComputeNewPrefixSum<<<gDim_PS2, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 0, 1024);
    if(eNum>1024*1024)ComputeNewPrefixSum<<<gDim_PS3, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 0, 1024*1024);
    if(eNum>1024*1024*1024)ComputeNewPrefixSum<<<gDim_PS4, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 0, 1024*1024*1024);
    if(eNum>1024){
       ComputeNewPrefixSum<<<gDim_PS, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 1, 1);
       ComputeNewPrefixSum<<<gDim_PS, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 2, 1);
       ComputeNewPrefixSum<<<gDim_PS, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 3, 1);
    }
    GPUSync("ERROR (Prefixsum):");
    return;
}
//==============================================================================


//================================== PrefixSum ==================================
void PrefixSum(const int eNum, int* xVector, int* yVector, int** xPSVector, int** yPSVector, char isNew, char psDim){
    int* yPSVect;
    if(isNew){
      cudaError_t cudaMemError=cudaMalloc((void**)xPSVector, sizeof(int)*eNum);
      GPUMAllocCheck(cudaMemError, "xPSVector");
      if(psDim==2){
         cudaMemError=cudaMalloc((void**)yPSVector, sizeof(int)*eNum);
         GPUMAllocCheck(cudaMemError, "yPSVector");
      }
       else yPSVect=NULL;
 
     }
     else yPSVect=*yPSVector;
    int blk=sqrt(eNum/1000)+1, blk2=sqrt(eNum/1000000)+1, blk3=sqrt(eNum/1000000000)+1, blk4=sqrt(eNum/1000000000000)+1;
    dim3 bDim_PS(1024, 1, 1), gDim_PS(blk, blk, psDim); 
    dim3 gDim_PS2(blk2, blk2, psDim), gDim_PS3(blk3, blk3, psDim), gDim_PS4(blk4, blk4, psDim);    

    ComputeNewPrefixSum<<<gDim_PS, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 0, 1);
    if(eNum>1024)ComputeNewPrefixSum<<<gDim_PS2, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 0, 1024);
    if(eNum>1024*1024)ComputeNewPrefixSum<<<gDim_PS3, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 0, 1024*1024);
    if(eNum>1024*1024*1024)ComputeNewPrefixSum<<<gDim_PS4, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 0, 1024*1024*1024);
    if(eNum>1024){
       ComputeNewPrefixSum<<<gDim_PS, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 1, 1);
       ComputeNewPrefixSum<<<gDim_PS, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 2, 1);
       ComputeNewPrefixSum<<<gDim_PS, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 3, 1);
    }
    GPUSync("ERROR (Prefixsum):");
    return;
}
//==============================================================================


//================================= PrefixSum ==================================
void PrefixSum(const int eNum, int* xVector, int* yVector, long** xPSVector, long** yPSVector, char isNew, char psDim){
    long* yPSVect;
    if(isNew){
      cudaError_t cudaMemError=cudaMalloc((void**)xPSVector, sizeof(long)*eNum);
      GPUMAllocCheck(cudaMemError, "xPSVector");
      if(psDim==2){
         cudaMemError=cudaMalloc((void**)yPSVector, sizeof(long)*eNum);
         GPUMAllocCheck(cudaMemError, "yPSVector");
         yPSVect=*yPSVector;
      }
      else yPSVect=NULL;
     }
     else yPSVect=*yPSVector;

    dim3 bDim_PS(1024, 1, 1), gDim_PS(eNum/1000+1, psDim, 1);    
    dim3 gDim_PS2(eNum/1000000+1, psDim, 1), gDim_PS3(eNum/1000000000+1, psDim, 1);    

    ComputeNewPrefixSum<<<gDim_PS, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 0, 1);
    if(eNum>1024)ComputeNewPrefixSum<<<gDim_PS2, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 0, 1024);
    if(eNum>1024*1024)ComputeNewPrefixSum<<<gDim_PS3, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 0, 1024*1024);
    if(eNum>1024){
       ComputeNewPrefixSum<<<gDim_PS, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 1, 1);
       ComputeNewPrefixSum<<<gDim_PS, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 2, 1);
    }
    GPUSync("ERROR (Prefixsum):");
    return;
}
//==============================================================================



//================================= PrefixSum ==================================
void PrefixSum(const int eNum, unsigned int* xVector, unsigned int* yVector, unsigned int** xPSVector, unsigned int** yPSVector, char isNew, char psDim){
    unsigned int* yPSVect;
    if(isNew){
      cudaError_t cudaMemError=cudaMalloc((void**)xPSVector, sizeof(unsigned int)*eNum);
      GPUMAllocCheck(cudaMemError, "xPSVector");
      if(psDim==2){
         cudaMemError=cudaMalloc((void**)yPSVector, sizeof(unsigned int)*eNum);
         GPUMAllocCheck(cudaMemError, "yPSVector");
         yPSVect=*yPSVector;
      }
      else yPSVect=NULL;
     }
     else yPSVect=*yPSVector;

    dim3 bDim_PS(1024, 1, 1), gDim_PS(eNum/1000+1, psDim, 1);    
    dim3 gDim_PS2(eNum/1000000+1, psDim, 1), gDim_PS3(eNum/1000000000+1, psDim, 1);    

    ComputeNewPrefixSum<<<gDim_PS, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 0, 1);
    if(eNum>1024)ComputeNewPrefixSum<<<gDim_PS2, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 0, 1024);
    if(eNum>1024*1024)ComputeNewPrefixSum<<<gDim_PS3, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 0, 1024*1024);
    if(eNum>1024){
       ComputeNewPrefixSum<<<gDim_PS, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 1, 1);
       ComputeNewPrefixSum<<<gDim_PS, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 2, 1);
    }
    GPUSync("ERROR (Prefixsum):");
    return;
}
//==============================================================================



//================================= PrefixSum ==================================
void PrefixSum(const int eNum, unsigned int* xVector, unsigned int* yVector, long** xPSVector, long** yPSVector, char isNew, char psDim){
    long* yPSVect;
    if(isNew){
      cudaError_t cudaMemError=cudaMalloc((void**)xPSVector, sizeof(long)*eNum);
      GPUMAllocCheck(cudaMemError, "xPSVector");
      if(psDim==2){
         cudaMemError=cudaMalloc((void**)yPSVector, sizeof(long)*eNum);
         GPUMAllocCheck(cudaMemError, "yPSVector");
         yPSVect=*yPSVector;
      }
      else yPSVect=NULL;
     }
     else yPSVect=*yPSVector;

    dim3 bDim_PS(1024, 1, 1), gDim_PS(eNum/1000+1, psDim, 1);    
    dim3 gDim_PS2(eNum/1000000+1, psDim, 1), gDim_PS3(eNum/1000000000+1, psDim, 1), gDim_PS4(eNum/1000000000000+1, psDim, 1);    

    ComputeNewPrefixSum<<<gDim_PS, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 0, 1);
    if(eNum>1024)ComputeNewPrefixSum<<<gDim_PS2, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 0, 1024);
    if(eNum>1024*1024)ComputeNewPrefixSum<<<gDim_PS3, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 0, 1024*1024);
    if(eNum>1024*1024*1024)ComputeNewPrefixSum<<<gDim_PS4, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 0, 1024*1024*1024);
    if(eNum>1024){
       ComputeNewPrefixSum<<<gDim_PS, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 1, 1);
       ComputeNewPrefixSum<<<gDim_PS, bDim_PS>>>(eNum, xVector, yVector, *xPSVector, yPSVect, 2, 1);
    }
    GPUSync("ERROR (Prefixsum):");
    return;
}
//==============================================================================


//=========================== ComputeNewPrefixSum ==============================
void BucketPrefixSum(const int elementNum, int* xIndex, int* yIndex, int* xVector, int* yVector, int* xPSVector, int* yPSVector, int bucketWidth, int bucketIndx, char psDim){
    dim3 bDim_DBPS(1024,1,1), gDim_DBPS(elementNum/1000+1, 10, psDim);    
    dim3 gDim_DBPS2(elementNum/(1000*1000)+1, 10, psDim), gDim_DBPS3(elementNum/(1000*1000*1000)+1, 10, psDim);    

    ComputeBucketPrefixSum<<<gDim_DBPS, bDim_DBPS, 0>>>(elementNum, xIndex, yIndex, xVector, yVector, xPSVector, yPSVector, bucketIndx, 0,1);
    if(elementNum>1024)ComputeBucketPrefixSum<<<gDim_DBPS, bDim_DBPS, 0>>>(elementNum, xIndex, yIndex, xVector, yVector, xPSVector, yPSVector, bucketIndx, 0,1024);
    if(elementNum>1024*1024)ComputeBucketPrefixSum<<<gDim_DBPS, bDim_DBPS, 0>>>(elementNum, xIndex, yIndex, xVector, yVector, xPSVector, yPSVector, bucketIndx, 0,1024*1024);
    if(elementNum>1024){
       ComputeBucketPrefixSum<<<gDim_DBPS, bDim_DBPS, 0>>>(elementNum, xIndex, yIndex, xVector, yVector, xPSVector, yPSVector, bucketIndx, 1,1);
       ComputeBucketPrefixSum<<<gDim_DBPS, bDim_DBPS, 0>>>(elementNum, xIndex, yIndex, xVector, yVector, xPSVector, yPSVector, bucketIndx, 2,1);
    }
    GPUSync("ERROR (BucketPrefixsum):");
}

//==============================================================================


//================================ RadixSort ===================================
void RadixSort(mbr_t* dxMBR, mbr_t* dyMBR, int *dxMBRIndex, int* dyMBRIndex, int* dxSortIndex, int* dySortIndex, int* dxSortIndex2, int* dySortIndex2, int lowerDigit, int upperDigit, int digitNum, long elementNum, int dimSort){
    //If dimSort=1, the code sorts just X dimension. If dimSort=2, both dimensions are sorted.
    cudaError_t cudaMemError;
    int *djCounter2, *djVector2, *dxDigitCounter, *dyDigitCounter, *dxPSDigitBucket, *dyPSDigitBucket; 
    int *dxPSDigitCounter, *dyPSDigitCounter; 
    cudaMemError=cudaMalloc((void**)&dxDigitCounter,10*MAX_DIGITS*sizeof(int)*elementNum);    
    GPUMAllocCheck(cudaMemError, "dxDigitCounter");
    cudaMemError=cudaMalloc((void**)&dxPSDigitCounter,10*sizeof(int)*elementNum);    
    GPUMAllocCheck(cudaMemError, "dxPSDigitCounter");
    cudaMemError=cudaMalloc((void**)&dxPSDigitBucket,10*sizeof(int));    
    GPUMAllocCheck(cudaMemError, "dxPSDigitBucket");
    if(dimSort==2){
      cudaMemError=cudaMalloc((void**)&dyDigitCounter,10*MAX_DIGITS*sizeof(int)*elementNum);    
      GPUMAllocCheck(cudaMemError, "dyDigitCounter");
      cudaMemError=cudaMalloc((void**)&dyPSDigitCounter,10*sizeof(int)*elementNum);    
      GPUMAllocCheck(cudaMemError, "dyPSDigitCounter");
      cudaMemError=cudaMalloc((void**)&dyPSDigitBucket,10*sizeof(int));    
      GPUMAllocCheck(cudaMemError, "dyPSDigitBucket");
    }
    dim3 bDim_DigCount(10, MAX_DIGITS, 1), gDim_DigCount((elementNum/1000)+1, 1000, 1);        
    ComputeDigitBuckets2<<<gDim_DigCount, bDim_DigCount, 0>>>(elementNum, dxMBR, dyMBR, dxDigitCounter, dyDigitCounter, dxSortIndex, dySortIndex, dxMBRIndex, dyMBRIndex, dimSort);
    GPUSync("ERROR (ComputeDigitBuckets2)");
    dim3 bDim_DBSort(128,1,1), gDim_DBSort(elementNum/100+1, dimSort, 1);    
    mbr_t digPow=1;
    for(int i=lowerDigit;i<=upperDigit;i++){
        if(i%2==0){
           BucketPrefixSum(elementNum, dxSortIndex, dySortIndex, dxDigitCounter, dyDigitCounter, dxPSDigitCounter, dyPSDigitCounter, 10, i, 1);
    	   //TestPSDigitCounter(xDigitCounter, xPSDigitCounter, xSortIndex, elementNum, i);
	   DigitBaseSort<<<gDim_DBSort, bDim_DBSort>>>(elementNum, dxMBR, dyMBR, dxSortIndex, dySortIndex, dxSortIndex2, dySortIndex2, dxMBRIndex, dyMBRIndex, dxDigitCounter, dyDigitCounter, dxPSDigitCounter, dyPSDigitCounter, digPow, i==upperDigit);
           GPUSync("ERROR (DigitBaseSort)");
        }
        else{
	   BucketPrefixSum(elementNum, dxSortIndex2, dySortIndex2, dxDigitCounter, dyDigitCounter, dxPSDigitCounter, dyPSDigitCounter, 10, i, 1);
	   DigitBaseSort<<<gDim_DBSort, bDim_DBSort>>>(elementNum, dxMBR, dyMBR, dxSortIndex2, dySortIndex2, dxSortIndex, dySortIndex, dxMBRIndex, dyMBRIndex, dxDigitCounter, dyDigitCounter, dxPSDigitCounter, dyPSDigitCounter, digPow, i==upperDigit);
           GPUSync("ERROR (DigitBaseSort)");
         }
         digPow*=10;
    }
    cudaFree(dxDigitCounter);
    cudaFree(dxPSDigitCounter);
    cudaFree(dxPSDigitBucket);
    if(dimSort==2){
      cudaFree(dyPSDigitCounter);
      cudaFree(dyDigitCounter);
      cudaFree(dyPSDigitBucket);
    }
    return;
}
//==============================================================================
