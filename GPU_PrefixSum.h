
//====================================================== ComputeBucketPrefixSum ====================================================================
//This set of GPU kernel launches are used to calculate prefix sum over two independent structure of 10 integers of the same size. 
//................................Parameters.....................................
//eNum				: Size of vectors
//xSortIndex, yIndexSort	: Indices of corresponding input vectors. Inputs are access through these indices (and not original order).
//xBucketVector, yBucketVector		: input vectors
//xPSBucketVector, yPSBucketVector	: output vectors of their corresponding input
//digitPos				: Identifies which of 10 intergers should be considered for prefix sum.
//pass, elementDistance		: They are algorithmic parameters.
//..............................................................................

//---------------------------------------------------ComputeBucketPrefixSum (inputs: int , outputs: int)--------------------------------------------
__global__ void ComputeBucketPrefixSum(const long eNum, int* xSortIndex, int* ySortIndex, int* xBucketVector, int* yBucketVector, int* xPSBucketVector, int* yPSBucketVector, char digitPos, char pass, int elementDistance){
    __shared__ long pSum[1024];
    __shared__ long preSum, preSum2;

    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y, bZIndx=blockIdx.z, digitVal=bYIndx;
    long elementIndx=bXIndx*blockDim.x+tXIndx;
    elementIndx=(elementIndx+1)*elementDistance-1;

    if(bYIndx>=10 || elementIndx>=eNum)return;

    int* vector, *sortIndex;
    int* prefixSum, gIndx, gPSIndx;

    if(bZIndx==0){        
        vector=xBucketVector;
        prefixSum=xPSBucketVector;
        sortIndex=xSortIndex;
    }
    else if(bZIndx==1){
        vector=yBucketVector;
        prefixSum=yPSBucketVector;
        sortIndex=ySortIndex;
    }        
    else{
	return;
    }
    long sortedElementIndx=eNum;
    if(elementIndx<eNum)sortedElementIndx=*(sortIndex+elementIndx);
    gIndx=sortedElementIndx*MAX_DIGITS*10+digitPos*10+digitVal;
    gPSIndx=elementIndx*10+digitVal;

    if(pass==0){
        int p=1;
        if(sortedElementIndx<eNum){
            if(elementDistance==1){
              pSum[tXIndx]=vector[gIndx];
            }
            else
              pSum[tXIndx]=prefixSum[gPSIndx];
        }
        else{
            pSum[tXIndx]=0;
        }

        __syncthreads();

        long tempSum;
        while(p<blockDim.x){        
            if(tXIndx-p>=0 && (sortedElementIndx<eNum)){
                tempSum=pSum[tXIndx]+pSum[tXIndx-p];
            }                    

            __syncthreads();

            if(tXIndx-p>=0 && (sortedElementIndx<eNum)){
                pSum[tXIndx]=tempSum;
            }
            __syncthreads();
            
            p*=2;
        }

        if(sortedElementIndx<eNum){
            *(prefixSum+gPSIndx)=pSum[tXIndx];
        }        
    }
    else if(pass==1){                
        if(tXIndx==0){
           int bXIndx2=bXIndx/blockDim.x;
	   long  oneBlockShift=blockDim.x*10, shiftBackAdjustment=(10-digitVal);
           preSum=0, preSum2=0;
	   if(0<bXIndx && bXIndx*blockDim.x<eNum){
               //If it is not very first block
	       preSum=*(prefixSum + bXIndx*oneBlockShift-shiftBackAdjustment);
           }
           if(0<bXIndx2 && bXIndx*blockDim.x<eNum){
              //If it is not very first big block chunk
              long oneBlock2Shift=blockDim.x*blockDim.x*10;
              if(bXIndx2*blockDim.x<bXIndx)
                 //If it is not first block at the big block chunk
                 preSum2=*(prefixSum + bXIndx2*oneBlock2Shift-shiftBackAdjustment);
           }
        }
      
        __syncthreads();
        
        if(sortedElementIndx<eNum && (tXIndx<blockDim.x-1)) 
            *(prefixSum+gPSIndx)+=preSum+preSum2;

    }
    else if(pass==2){                
        if(tXIndx==0){
           int bXIndx2=bXIndx/blockDim.x;
	   long  oneBlockShift=blockDim.x*10, shiftBackAdjustment=(10-digitVal);
           preSum=0, preSum2=0;
           if(0<bXIndx2 && bXIndx*blockDim.x<eNum){
              //If it is not very first big block chunk
              long oneBlock2Shift=blockDim.x*blockDim.x*10;
              if( (bXIndx+1)%blockDim.x!=0)
                 //If it is not the last block at the big block chunk
                 preSum2=*(prefixSum + bXIndx2*oneBlock2Shift-shiftBackAdjustment);
           }
        }
      
        __syncthreads();
        
        if(sortedElementIndx<eNum && (tXIndx==blockDim.x-1))
            *(prefixSum+gPSIndx)+=preSum+preSum2;
    }

   return;
}
//================================================================================================================================================


//======================================================== ComputeNewPrefixSum ===================================================================
//This set of GPU kernel launches are used to calculate prefix sum over two independent vectors of the same size. 
//................................Parameters.....................................
//eNum				: Size of vectors
//xVector, yVector		: input vectors
//xPSVector, yPSVector		: output vectors of their corresponding input
//pass, elementDistance		: They are algorithmic parameters.
//..............................................................................

//---------------------------------------------------ComputeNewPrefixSum (inputs: unsinged int , outputs: long)--------------------------------------------
__global__ void ComputeNewPrefixSum(const int eNum, unsigned int* xVector, unsigned int* yVector, unsigned int* xPSVector, unsigned int* yPSVector, char pass, int elementDistance){
    __shared__ unsigned int pSum[1024];
    __shared__ unsigned int preSum, preSum2;

    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y;
    long elementIndx=bXIndx*blockDim.x+tXIndx;
    elementIndx=(elementIndx+1)*elementDistance-1;
    if(elementIndx>=eNum)return;

    unsigned int* vector;
    unsigned int* prefixSum;

    if(bYIndx==0){        
        vector=xVector;
        prefixSum=xPSVector;
    }
    else if(bYIndx==1){
        vector=yVector;
        prefixSum=yPSVector;
    }        
    else{
	return;
    }

    if(pass==0){
        int p=1;
        if(elementIndx<eNum){
            if(elementDistance==1){
              pSum[tXIndx]=vector[elementIndx];
            }
            else
              pSum[tXIndx]=prefixSum[elementIndx];
        }
        else{
            pSum[tXIndx]=0;
        }

        __syncthreads();

        unsigned int tempSum;
        while(p<blockDim.x){        
            if(tXIndx-p>=0 && (elementIndx<eNum)){
                tempSum=pSum[tXIndx]+pSum[tXIndx-p];
            }                    

            __syncthreads();

            if(tXIndx-p>=0 && (elementIndx<eNum)){
                pSum[tXIndx]=tempSum;
            }
            __syncthreads();
            p*=2;
        }

        if(elementIndx<eNum){
            *(prefixSum+elementIndx)=pSum[tXIndx];
        }        
    }
    else if(pass==1){                
        if(tXIndx==0){
           int bXIndx2=bXIndx/blockDim.x;
	   long  oneBlockShift=blockDim.x;
           preSum=0, preSum2=0;
	   if(0<bXIndx && bXIndx*blockDim.x<eNum){
               //If it is not very first block
	       preSum=*(prefixSum + bXIndx*oneBlockShift-1);
           }
           if(0<bXIndx2 && bXIndx*blockDim.x<eNum){
              //If it is not very first big block chunk
              long oneBlock2Shift=blockDim.x*blockDim.x;
              if(bXIndx2*blockDim.x<bXIndx)
                 //If it is not first block at the big block chunk
                 preSum2=*(prefixSum + bXIndx2*oneBlock2Shift-1);
           }
        }
      
        __syncthreads();
        
        if(elementIndx<eNum && (tXIndx<blockDim.x-1)) 
            *(prefixSum+elementIndx)+=preSum+preSum2;

    }
    else if(pass==2){                
        if(tXIndx==0){
           int bXIndx2=bXIndx/blockDim.x;
	   long  oneBlockShift=blockDim.x;
           preSum=0, preSum2=0;
           if(0<bXIndx2 && bXIndx*blockDim.x<eNum){
              //If it is not very first big block chunk
              long oneBlock2Shift=blockDim.x*blockDim.x;
              if( (bXIndx+1)%blockDim.x!=0)
                 //If it is not the last block at the big block chunk
                 preSum2=*(prefixSum + bXIndx2*oneBlock2Shift-1);
           }
        }
      
        __syncthreads();
        
        if(elementIndx<eNum && (tXIndx==blockDim.x-1))
            *(prefixSum+elementIndx)+=preSum+preSum2;
    }

   return;
}
//-----------------------------------------------------------------------------------------------------------------------------------------------


//---------------------------------------------------ComputeNewPrefixSum (inputs: unsinged int , outputs: long)--------------------------------------------
__global__ void ComputeNewPrefixSum(const int eNum, unsigned int* xVector, unsigned int* yVector, long* xPSVector, long* yPSVector, char pass, int elementDistance){
    __shared__ long pSum[1024];
    __shared__ long preSum, preSum2;

    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y;
    long elementIndx=bXIndx*blockDim.x+tXIndx;
    elementIndx=(elementIndx+1)*elementDistance-1;
    if(elementIndx>=eNum)return;

    unsigned int* vector;
    long* prefixSum;

    if(bYIndx==0){        
        vector=xVector;
        prefixSum=xPSVector;
    }
    else if(bYIndx==1){
        vector=yVector;
        prefixSum=yPSVector;
    }        
    else{
	return;
    }

    if(pass==0){
        int p=1;
        if(elementIndx<eNum){
            if(elementDistance==1){
              pSum[tXIndx]=vector[elementIndx];
            }
            else
              pSum[tXIndx]=prefixSum[elementIndx];
        }
        else{
            pSum[tXIndx]=0;
        }

        __syncthreads();

        long tempSum;
        while(p<blockDim.x){        
            if(tXIndx-p>=0 && (elementIndx<eNum)){
                tempSum=pSum[tXIndx]+pSum[tXIndx-p];
            }                    

            __syncthreads();

            if(tXIndx-p>=0 && (elementIndx<eNum)){
                pSum[tXIndx]=tempSum;
            }
            __syncthreads();
            p*=2;
        }

        if(elementIndx<eNum){
            *(prefixSum+elementIndx)=pSum[tXIndx];
        }        
    }
    else if(pass==1){                
        if(tXIndx==0){
           int bXIndx2=bXIndx/blockDim.x;
	   long  oneBlockShift=blockDim.x;
           preSum=0, preSum2=0;
	   if(0<bXIndx && bXIndx*blockDim.x<eNum){
               //If it is not very first block
	       preSum=*(prefixSum + bXIndx*oneBlockShift-1);
           }
           if(0<bXIndx2 && bXIndx*blockDim.x<eNum){
              //If it is not very first big block chunk
              long oneBlock2Shift=blockDim.x*blockDim.x;
              if(bXIndx2*blockDim.x<bXIndx)
                 //If it is not first block at the big block chunk
                 preSum2=*(prefixSum + bXIndx2*oneBlock2Shift-1);
           }
        }
      
        __syncthreads();
        
        if(elementIndx<eNum && (tXIndx<blockDim.x-1)) 
            *(prefixSum+elementIndx)+=preSum+preSum2;

    }
    else if(pass==2){                
        if(tXIndx==0){
           int bXIndx2=bXIndx/blockDim.x;
	   long  oneBlockShift=blockDim.x;
           preSum=0, preSum2=0;
           if(0<bXIndx2 && bXIndx*blockDim.x<eNum){
              //If it is not very first big block chunk
              long oneBlock2Shift=blockDim.x*blockDim.x;
              if( (bXIndx+1)%blockDim.x!=0)
                 //If it is not the last block at the big block chunk
                 preSum2=*(prefixSum + bXIndx2*oneBlock2Shift-1);
           }
        }
      
        __syncthreads();
        
        if(elementIndx<eNum && (tXIndx==blockDim.x-1))
            *(prefixSum+elementIndx)+=preSum+preSum2;
    }

   return;
}
//-----------------------------------------------------------------------------------------------------------------------------------------------


//---------------------------------------------------ComputeNewPrefixSum (inputs: int , outputs: long)--------------------------------------------
__global__ void ComputeNewPrefixSum(const int eNum, int* xVector, int* yVector, long* xPSVector, long* yPSVector, char pass, int elementDistance){
    __shared__ long pSum[1024];
    __shared__ long preSum, preSum2;

    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y;
    long elementIndx=bXIndx*blockDim.x+tXIndx;
    elementIndx=(elementIndx+1)*elementDistance-1;
    if(elementIndx>=eNum)return;

    int* vector;
    long* prefixSum;

    if(bYIndx==0){        
        vector=xVector;
        prefixSum=xPSVector;
    }
    else if(bYIndx==1){
        vector=yVector;
        prefixSum=yPSVector;
    }        
    else{
	return;
    }

    if(pass==0){
        int p=1;
        if(elementIndx<eNum){
            if(elementDistance==1){
              pSum[tXIndx]=vector[elementIndx];
            }
            else
              pSum[tXIndx]=prefixSum[elementIndx];
        }
        else{
            pSum[tXIndx]=0;
        }

        __syncthreads();

        long tempSum;
        while(p<blockDim.x){        
            if(tXIndx-p>=0 && (elementIndx<eNum)){
                tempSum=pSum[tXIndx]+pSum[tXIndx-p];
            }                    

            __syncthreads();

            if(tXIndx-p>=0 && (elementIndx<eNum)){
                pSum[tXIndx]=tempSum;
            }
            __syncthreads();
            p*=2;
        }

        if(elementIndx<eNum){
            *(prefixSum+elementIndx)=pSum[tXIndx];
        }        
    }
    else if(pass==1){                
        if(tXIndx==0){
           int bXIndx2=bXIndx/blockDim.x;
	   long  oneBlockShift=blockDim.x;
           preSum=0, preSum2=0;
	   if(0<bXIndx && bXIndx*blockDim.x<eNum){
               //If it is not very first block
	       preSum=*(prefixSum + bXIndx*oneBlockShift-1);
           }
           if(0<bXIndx2 && bXIndx*blockDim.x<eNum){
              //If it is not very first big block chunk
              long oneBlock2Shift=blockDim.x*blockDim.x;
              if(bXIndx2*blockDim.x<bXIndx)
                 //If it is not first block at the big block chunk
                 preSum2=*(prefixSum + bXIndx2*oneBlock2Shift-1);
           }
        }
      
        __syncthreads();
        
        if(elementIndx<eNum && (tXIndx<blockDim.x-1)) 
            *(prefixSum+elementIndx)+=preSum+preSum2;

    }
    else if(pass==2){                
        if(tXIndx==0){
           int bXIndx2=bXIndx/blockDim.x;
	   long  oneBlockShift=blockDim.x;
           preSum=0, preSum2=0;
           if(0<bXIndx2 && bXIndx*blockDim.x<eNum){
              //If it is not very first big block chunk
              long oneBlock2Shift=blockDim.x*blockDim.x;
              if( (bXIndx+1)%blockDim.x!=0)
                 //If it is not the last block at the big block chunk
                 preSum2=*(prefixSum + bXIndx2*oneBlock2Shift-1);
           }
        }
      
        __syncthreads();
        
        if(elementIndx<eNum && (tXIndx==blockDim.x-1))
            *(prefixSum+elementIndx)+=preSum+preSum2;
    }

   return;
}
//-----------------------------------------------------------------------------------------------------------------------------------------------


//---------------------------------------------------ComputeNewPrefixSum (inputs: int , outputs: int)--------------------------------------------
__global__ void ComputeNewPrefixSum(const int eNum, char* xVector, char* yVector, long* xPSVector, long* yPSVector, char pass, int elementDistance){
    __shared__ int pSum[1024];
    __shared__ int preSum, preSum2, preSum3;

    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y, bZIndx=blockIdx.z;
    long elementIndx = bYIndx * gridDim.x * blockDim.x + bXIndx * blockDim.x + tXIndx, bIndx=bYIndx*gridDim.x+bXIndx;
    elementIndx=(elementIndx+1)*elementDistance-1;
    if(elementIndx>=eNum)return;
    long  *prefixSum;
    char* vector;

    if(bZIndx==0){        
        vector=xVector;
        prefixSum=xPSVector;
    }
    else if(bZIndx==1){
        vector=yVector;
        prefixSum=yPSVector;
    }        
    else{
	return;
    }

    if(pass==0){
        int p=1;
        if(elementIndx<eNum){
            if(elementDistance==1){
              pSum[tXIndx]=vector[elementIndx];
            }
            else
              pSum[tXIndx]=prefixSum[elementIndx];
        }
        else{
            pSum[tXIndx]=0;
        }

        __syncthreads();

        long tempSum;
        while(p<blockDim.x){        
            if(tXIndx-p>=0 && (elementIndx<eNum)){
                tempSum=pSum[tXIndx]+pSum[tXIndx-p];
            }                    

            __syncthreads();

            if(tXIndx-p>=0 && (elementIndx<eNum)){
                pSum[tXIndx]=tempSum;
            }
            __syncthreads();
            p*=2;
        }
        if(elementIndx<eNum){
            prefixSum[elementIndx]=pSum[tXIndx];
        }        
    }
    else if(pass==1){                
        if(tXIndx==0){
           int bXIndx2=bIndx/blockDim.x, bXIndx3=bIndx/(blockDim.x*blockDim.x);
	   long  oneBlockShift=blockDim.x, oneBlock2Shift=blockDim.x*blockDim.x, oneBlock3Shift=blockDim.x*blockDim.x*blockDim.x;
           preSum=0; preSum2=0; preSum3=0;
           if(bXIndx2*blockDim.x<bIndx){
           //If it is not first block in big block
	       preSum=*(prefixSum + bIndx*oneBlockShift-1);
           }
           if(bXIndx3*blockDim.x<bXIndx2){
           //If it is not first big block in big big block
              preSum2=*(prefixSum + bXIndx2*oneBlock2Shift-1);
           }
           if(0<bXIndx3){
           //If it is not very first big big block 
              preSum3=*(prefixSum + bXIndx3*oneBlock3Shift-1);
           }
       }
      
        __syncthreads();
        
        //Updating all the elements expepts last-block-elements
        if(elementIndx<eNum && (tXIndx<blockDim.x-1)) 
            *(prefixSum+elementIndx)+=preSum+preSum2+preSum3;

    }
    else if(pass==2){                
        int bXIndx2=bIndx/blockDim.x, bXIndx3=bIndx/(blockDim.x*blockDim.x);
        long oneBlock2Shift=blockDim.x*blockDim.x, oneBlock3Shift=blockDim.x*blockDim.x*blockDim.x;
        if(tXIndx==0){
           preSum2=0; preSum3=0;
           //If it is not first big block in big big block
           if(bXIndx3*blockDim.x<bXIndx2)preSum2=*(prefixSum + bXIndx2*oneBlock2Shift-1);
           //If it is not very first big big block
           if(0<bXIndx3)preSum3=*(prefixSum + bXIndx3*oneBlock3Shift-1);
        }
      
        __syncthreads();
       
        //Updating all last block elements expept last-block-elements of big blocks 
        if(elementIndx<eNum && (tXIndx==blockDim.x-1) && (bIndx+1)%blockDim.x!=0 )
            *(prefixSum+elementIndx)+=preSum2+preSum3;

    }
    else if(pass==3){                
        if(tXIndx==0){
           preSum3=0;
           long oneBlock2Shift=blockDim.x*blockDim.x, oneBlock3Shift=blockDim.x*blockDim.x*blockDim.x;
           int bXIndx2=bIndx/blockDim.x, bXIndx3=bIndx/(blockDim.x*blockDim.x);
           //If it is not very first big big block
           if(0<bXIndx3 &&  (bIndx+1)%blockDim.x==0 ) preSum3=*(prefixSum + bXIndx3*oneBlock3Shift-1);
        }
      
        __syncthreads();
        
        //Updating all last-block-elements of big block 
        if(elementIndx<eNum && (tXIndx==blockDim.x-1))
            *(prefixSum+elementIndx)+=preSum3;
    }


   return;
}
//----------------------------------------------------------------------------------------------------------------------------------------------


//---------------------------------------------------ComputeNewPrefixSum (inputs: int , outputs: int)--------------------------------------------
__global__ void ComputeNewPrefixSum(const int eNum, int* xVector, int* yVector, int* xPSVector, int* yPSVector, char pass, int elementDistance){
    __shared__ int pSum[1024];
    __shared__ int preSum, preSum2, preSum3;

    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y, bZIndx=blockIdx.z;
    long elementIndx = bYIndx * gridDim.x * blockDim.x + bXIndx * blockDim.x + tXIndx, bIndx=bYIndx*gridDim.x+bXIndx;
    elementIndx=(elementIndx+1)*elementDistance-1;
    if(elementIndx>=eNum)return;
    int* vector, *prefixSum;

    if(bZIndx==0){        
        vector=xVector;
        prefixSum=xPSVector;
    }
    else if(bZIndx==1){
        vector=yVector;
        prefixSum=yPSVector;
    }        
    else{
	return;
    }

    if(pass==0){
        int p=1;
        if(elementIndx<eNum){
            if(elementDistance==1){
              pSum[tXIndx]=vector[elementIndx];
            }
            else
              pSum[tXIndx]=prefixSum[elementIndx];
        }
        else{
            pSum[tXIndx]=0;
        }

        __syncthreads();

        long tempSum;
        while(p<blockDim.x){        
            if(tXIndx-p>=0 && (elementIndx<eNum)){
                tempSum=pSum[tXIndx]+pSum[tXIndx-p];
            }                    

            __syncthreads();

            if(tXIndx-p>=0 && (elementIndx<eNum)){
                pSum[tXIndx]=tempSum;
            }
            __syncthreads();
            p*=2;
        }
        if(elementIndx<eNum){
            prefixSum[elementIndx]=pSum[tXIndx];
        }        
    }
    else if(pass==1){                
        if(tXIndx==0){
           int bXIndx2=bIndx/blockDim.x, bXIndx3=bIndx/(blockDim.x*blockDim.x);
	   long  oneBlockShift=blockDim.x, oneBlock2Shift=blockDim.x*blockDim.x, oneBlock3Shift=blockDim.x*blockDim.x*blockDim.x;
           preSum=0; preSum2=0; preSum3=0;
           if(bXIndx2*blockDim.x<bIndx){
           //If it is not first block in big block
	       preSum=*(prefixSum + bIndx*oneBlockShift-1);
           }
           if(bXIndx3*blockDim.x<bXIndx2){
           //If it is not first big block in big big block
              preSum2=*(prefixSum + bXIndx2*oneBlock2Shift-1);
           }
           if(0<bXIndx3){
           //If it is not very first big big block 
              preSum3=*(prefixSum + bXIndx3*oneBlock3Shift-1);
           }
       }
      
        __syncthreads();
        
        //Updating all the elements expepts last-block-elements
        if(elementIndx<eNum && (tXIndx<blockDim.x-1)) 
            *(prefixSum+elementIndx)+=preSum+preSum2+preSum3;

    }
    else if(pass==2){                
        int bXIndx2=bIndx/blockDim.x, bXIndx3=bIndx/(blockDim.x*blockDim.x);
        long oneBlock2Shift=blockDim.x*blockDim.x, oneBlock3Shift=blockDim.x*blockDim.x*blockDim.x;
        if(tXIndx==0){
           preSum2=0; preSum3=0;
           //If it is not first big block in big big block
           if(bXIndx3*blockDim.x<bXIndx2)preSum2=*(prefixSum + bXIndx2*oneBlock2Shift-1);
           //If it is not very first big big block
           if(0<bXIndx3)preSum3=*(prefixSum + bXIndx3*oneBlock3Shift-1);
        }
      
        __syncthreads();
       
        //Updating all last block elements expept last-block-elements of big blocks 
        if(elementIndx<eNum && (tXIndx==blockDim.x-1) && (bIndx+1)%blockDim.x!=0 )
            *(prefixSum+elementIndx)+=preSum2+preSum3;

    }
    else if(pass==3){                
        if(tXIndx==0){
           preSum3=0;
           long oneBlock2Shift=blockDim.x*blockDim.x, oneBlock3Shift=blockDim.x*blockDim.x*blockDim.x;
           int bXIndx2=bIndx/blockDim.x, bXIndx3=bIndx/(blockDim.x*blockDim.x);
           //If it is not very first big big block
           if(0<bXIndx3 &&  (bIndx+1)%blockDim.x==0 ) preSum3=*(prefixSum + bXIndx3*oneBlock3Shift-1);
        }
      
        __syncthreads();
        
        //Updating all last-block-elements of big block 
        if(elementIndx<eNum && (tXIndx==blockDim.x-1))
            *(prefixSum+elementIndx)+=preSum3;
    }


   return;
}
//----------------------------------------------------------------------------------------------------------------------------------------------

//==============================================================================================================================================
