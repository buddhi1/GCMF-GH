#include "Types.h"

//----------------------------- PrintGPUVector ----------------------------------
void GPUPrintVector(int eNum, char* gpuVector, char suppressZero ){
   char *vector; 
   CopyFromGPU((void**)&vector, gpuVector, sizeof(char)*eNum, 1);
   for(int i=0;i<eNum;i++){
      if(suppressZero==1 && vector[i]==0)continue;
      printf("Element %d: %d\n", i, vector[i]);
   }
   free(vector);
   return;
}

void GPUPrintVector(int eNum, int* gpuVector, char suppressZero ){
   int *vector; 
   CopyFromGPU((void**)&vector, gpuVector, sizeof(int)*eNum, 1);
   for(int i=0;i<eNum;i++){
      if(suppressZero==1 && vector[i]==0)continue;
      printf("Element %d: %d\n", i, vector[i]);
   }
   free(vector);
   return;
}

void GPUPrintVector(int eNum, long* gpuVector, char suppressZero){
   long *vector; 
   CopyFromGPU((void**)&vector, gpuVector, sizeof(long)*eNum, 1);
   for(int i=0;i<eNum;i++){
      if(suppressZero==1 && vector[i]==0)continue;
      printf("Element %d: %ld\n", i, vector[i]);
   }
   free(vector);
   return;
}

void GPUPrintVector(int eNum, poly_size_t* gpuVector, char suppressZero){
   poly_size_t *vector; 
   CopyFromGPU((void**)&vector, gpuVector, sizeof(poly_size_t)*eNum, 1);
   for(int i=0;i<eNum;i++){
      if(suppressZero==1 && vector[i]==0)continue;
      printf("Element %d: %d\n", i, vector[i]);
   }
   free(vector);
   return;
}


void GPUPrintVector(int eNum, mbr_t* gpuVector, char suppressZero){
   mbr_t *vector; 
   CopyFromGPU((void**)&vector, gpuVector, sizeof(mbr_t)*eNum, 1);
   for(int i=0;i<eNum;i++){
      if(suppressZero==1 && vector[i]==0)continue;
      printf("Element %d: %ld\n", i, vector[i]);
   }
   free(vector);
   return;
}

void GPUPrintVector(int eNum, coord_t* gpuVector, char suppressZero){
   coord_t *vector; 
   CopyFromGPU((void**)&vector, gpuVector, sizeof(coord_t)*eNum, 1);
   for(int i=0;i<eNum;i++){
      if(suppressZero==1 && vector[i]==0)continue;
      printf("Element %d: %g\n", i, vector[i]);
   }
   free(vector);
   return;
}


//-------------------------------------------------------------------------------

//========================== PrintPresumVectorTest =============================
void GPUPrefixsumTest(int* gpuVector, long* gpuPSVector, long eNum, int printAll){
    long pFSum=0, *pSVector;
    int *Vector;
    CopyFromGPU((void**)&pSVector, gpuPSVector, sizeof(long)*eNum, 1);
    CopyFromGPU((void**)&Vector, gpuVector, sizeof(int)*eNum, 1);
    for(long i=0;i<eNum;i++){
        pFSum+=Vector[i];
        if(Vector[i]==0)continue;
        if(printAll==0 && pFSum==pSVector[i])continue;
        printf("\nPrefix Sum up to index %ld of %ld is: (%d , %ld):%ld", i, eNum, Vector[i], pFSum, pSVector[i]);
        if(printAll==0)return;
    }
    printf("\nSum over all elements: %d\n", pFSum);
    printf("\nPrefix vector is successfully verified.\n");
}

void GPUPrefixsumTest(int* gpuVector, int* gpuPSVector, long eNum, int printAll){
    int pFSum=0, *pSVector, *Vector;
    CopyFromGPU((void**)&pSVector, gpuPSVector, sizeof(int)*eNum, 1);
    CopyFromGPU((void**)&Vector, gpuVector, sizeof(int)*eNum, 1);
    for(long i=0;i<eNum;i++){
        pFSum+=Vector[i];
        if(Vector[i]==0)continue;
        if(printAll==0 && pFSum==pSVector[i])continue;
        printf("\nPrefix Sum up to index %ld of %ld is: (%d , %d):%d", i, eNum, Vector[i], pFSum, pSVector[i]);
        if(printAll==0){
           free(pSVector);
           free(Vector);
           return;
        }
    }
    printf("\nSum over all elements: %d\n", pFSum);
    printf("\nPrefix vector is successfully verified.\n");
    free(pSVector);
    free(Vector);
    return;
}
//==============================================================================

//============================ IndexCoverageTest ===============================
void GPUIndexCoverageTest(long eNum, int* gpuSortIndex){
   int coverageTest[eNum], *sortIndex;    
   CopyFromGPU((void**)&sortIndex, gpuSortIndex, sizeof(int)*eNum, 1);
   bool flag=false;
   for(int k=0;k<eNum;k++){
      coverageTest[k]=0;
   }
   for(int k=0;k<eNum;k++){
      coverageTest[sortIndex[k]]=1;
   }
   for(int k=0;k<eNum;k++){
      if(coverageTest[k]==0){
         printf("\nUncovered index at : %d\n", k);
         flag=true;
         free(sortIndex);
         return;
      }
      //if(coverageTest[k]>1){printf("\nIndex at : %d  covered %d times", k, coverageTest[k]);flag=true;}
   }
   if(!flag)printf("\n\n Full coverage for index range [0-%d]", eNum);
   free(sortIndex);
   return;
}
//==============================================================================


//============================ GPUMBRIndexTest ===============================
void GPUMBRIndexTest(long eNum, int* gpuMBRIndex, int* gpuSortIndex, cmbr_t* gpuMBR){
   int mIndx, *MBRIndex, *sortIndex;
   cmbr_t *mbr;
   CopyFromGPU((void**)&sortIndex, gpuSortIndex, sizeof(int)*eNum, 1);
   CopyFromGPU((void**)&MBRIndex, gpuMBRIndex, sizeof(int)*eNum, 1);
   CopyFromGPU((void**)&mbr, gpuMBR, sizeof(cmbr_t)*eNum, 1);
   for(int k=0;k<eNum;k++){
      mIndx=MBRIndex[k];
      if(mIndx%2==1 && MBRIndex[k-1]>mIndx){
         int mIndx0=MBRIndex[k-1];
	 printf("\nMBR Index Error at MBR %d:  Left side: [%d]=%ld \t Right side: [%d]=%ld\n", k, mIndx0, mbr[k-1], mIndx, mbr[k]);
         free(MBRIndex);
         free(sortIndex);
         return;
      }
      if(sortIndex[mIndx]==k)continue;
      printf("\n(MBRIndex: %d): %d=?=%d", mIndx, k, sortIndex[mIndx]);
      free(MBRIndex);
      free(sortIndex);
      return;
   }
   printf("\nMBRIndex Successfully Verified for index range[0-%ld]!\n", eNum);
   free(MBRIndex);
   free(sortIndex);
   return;
}
//==============================================================================


//============================ GPUPrintSortedData ===============================
void GPUPrintSortedData(int eNum, cmbr_t* gpuVector, int* gpuSortIndex){
   cmbr_t* vector;
   int *sortIndex;
   CopyFromGPU((void**)&sortIndex, gpuSortIndex, sizeof(int)*eNum, 1);
   CopyFromGPU((void**)&vector, gpuVector, sizeof(cmbr_t)*eNum, 1);
   printf("\nPrinting out the sorted sequence...\n");
   for(int k=0;k<eNum;k++){
      printf("\n%d:\t\tElement[%d]:\t%ld", k, sortIndex[k], vector[sortIndex[k]]);
   }
   free(sortIndex);
   free(vector);
   return;
}
//==============================================================================





