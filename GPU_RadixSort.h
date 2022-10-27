

//=========================== ComputeDigitBuckets2 ==============================
__global__ void InitializeData(long bPolNum, long oPolNum, mbr_t *bMBR, mbr_t *oMBR,  cmbr_t* xMBR, cmbr_t* yMBR){
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y;        
    long blockIndx=bYIndx*gridDim.x+bXIndx;
    long gtIndx=blockIndx*blockDim.x+tXIndx;
    if(gtIndx>=(bPolNum+oPolNum))return;
    if(gtIndx<bPolNum){
       xMBR[gtIndx*2]=bMBR[gtIndx*4];
       xMBR[gtIndx*2+1]=bMBR[gtIndx*4+2];
       yMBR[gtIndx*2]=bMBR[gtIndx*4+1];
       yMBR[gtIndx*2+1]=bMBR[gtIndx*4+3];
    }
    else{
       xMBR[gtIndx*2]=oMBR[(gtIndx-bPolNum)*4];
       xMBR[gtIndx*2+1]=oMBR[(gtIndx-bPolNum)*4+2];
       yMBR[gtIndx*2]=oMBR[(gtIndx-bPolNum)*4+1];
       yMBR[gtIndx*2+1]=oMBR[(gtIndx-bPolNum)*4+3];
    }
    return;
}    
//==============================================================================


//=========================== ComputeDigitBuckets2 ==============================
__global__ void ComputeDigitBuckets2(long elementNum, cmbr_t* xMBR, cmbr_t* yMBR, int *xDigitCounter, int *yDigitCounter, int *xSortIndex, int* ySortIndex, int *xMBRIndex, int* yMBRIndex, int dimSort){
    __shared__ cmbr_t sxMBR, syMBR;
    int tXIndx=threadIdx.x, tYIndx=threadIdx.y, bXIndx=blockIdx.x, bYIndx=blockIdx.y;        
    long blockIndx;
    blockIndx=bYIndx*gridDim.x+bXIndx;
   if(blockIndx>=elementNum)return;

   if(tXIndx==0 && tYIndx==0){
      sxMBR=*(xMBR+blockIndx);
      *(xSortIndex+blockIndx)=blockIndx; 
      *(xMBRIndex+blockIndx)=blockIndx;
      if(dimSort==2){
        syMBR=*(yMBR+blockIndx);
        *(ySortIndex+blockIndx)=blockIndx; 
        *(yMBRIndex+blockIndx)=blockIndx;
      }
   }
   
   __syncthreads();

   if(tYIndx<MAX_DIGITS && tXIndx<10){
     cmbr_t digPos=0, pos=1, xDigit, yDigit;
     while(digPos++<tYIndx)pos*=10;

     xDigit=(sxMBR/pos)%10;
     *(xDigitCounter+blockIndx*MAX_DIGITS*10+tYIndx*10+tXIndx)=(tXIndx==xDigit?1:0);

     if(dimSort==2){
       yDigit=(syMBR/pos)%10;
       *(yDigitCounter+blockIndx*MAX_DIGITS*10+tYIndx*10+tXIndx)=(tXIndx==yDigit?1:0);
     }
   }
   return;
}    
//==============================================================================


//=========================== ComputeDigitBuckets ==============================
__global__ void ComputeDigitBuckets(long elementNum, cmbr_t* xMBR, cmbr_t* yMBR, int *xDigitCounter, int *yDigitCounter){
    __shared__ cmbr_t sxMBR, syMBR;
    int tXIndx=threadIdx.x, tYIndx=threadIdx.y, bXIndx=blockIdx.x, bYIndx=blockIdx.y;        
    long blockIndx=bYIndx*gridDim.x+bXIndx;
    cmbr_t pos=1;    

   if(blockIndx>=elementNum)return;

   if(tXIndx==0 && tYIndx==0){
      sxMBR=*(xMBR+blockIndx);
      syMBR=*(yMBR+blockIndx);
   }
  
   __syncthreads();

   if(tYIndx<MAX_DIGITS && tXIndx<10){
     int digPos=0, pos=1, xDigit, yDigit;
     while(digPos++<tYIndx)pos*=10;

     xDigit=(sxMBR/pos)%10;
     *(xDigitCounter+blockIndx*MAX_DIGITS*10+tYIndx*10+tXIndx)=(tXIndx==xDigit?1:0);

     yDigit=(syMBR/pos)%10;
     *(yDigitCounter+blockIndx*MAX_DIGITS*10+tYIndx*10+tXIndx)=(tXIndx==yDigit?1:0);
   }
   return;
}    
//==============================================================================


//=========================== DigitBaseSort ==============================
__global__ void DigitBaseSort(long eNum, cmbr_t* xMBR, cmbr_t* yMBR, int* xSortIndex, int* ySortIndex, int* xSortIndex2, int* ySortIndex2, int* xMBRIndex, int* yMBRIndex, int* xDigitCounter, int* yDigitCounter, int* xPSDigitCounter, int* yPSDigitCounter, cmbr_t digitPow, char isLastDigit){
    __shared__ cmbr_t *sMBR;
    __shared__ int* sDigitCounter, *sSortIndex, *sSortIndex2, *sMBRIndex;
    __shared__ int* sPSDigitCounter, sPSDigitBucket[10];

    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y, digitVal;
    int elementIndx;

    elementIndx=bXIndx*blockDim.x+tXIndx;

    if(elementIndx>=eNum)return;

    if(tXIndx==0){
       if(bYIndx==0){        
           sDigitCounter=xDigitCounter;
           sPSDigitCounter=xPSDigitCounter;
           sSortIndex=xSortIndex;
           sSortIndex2=xSortIndex2;
   	   sMBRIndex=xMBRIndex;
	   sMBR=xMBR;
       }
       else if(bYIndx==1){
           sDigitCounter=yDigitCounter;
           sPSDigitCounter=yPSDigitCounter;
           sSortIndex=ySortIndex;
           sSortIndex2=ySortIndex2;
	   sMBRIndex=yMBRIndex;
	   sMBR=yMBR;
       }
       int lastBlockShift=(eNum-1)*10, digitBucketSum=0;
       for(int i=0;i<10;i++){
        digitBucketSum+=*(sPSDigitCounter+lastBlockShift+i);
        sPSDigitBucket[i]=digitBucketSum;
      }         
    }

    __syncthreads();

    int actualIndx=*(sSortIndex+elementIndx), gPSIndx;
    digitVal=(*(sMBR+actualIndx)/digitPow)%10;
    gPSIndx=elementIndx*10+digitVal;
    int shift=0, shiftIndex=0;
    shiftIndex=*(sPSDigitCounter+gPSIndx)-1;

    if(digitVal>0)shift=*(sPSDigitBucket+digitVal-1);
    shiftIndex+=shift;

    *(sSortIndex2+shiftIndex)=actualIndx;
    if(isLastDigit)*(sMBRIndex+actualIndx)=shiftIndex;
    return;
  }
//==============================================================================

