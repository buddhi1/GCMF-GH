#include "Constants.h"
#include "Types.h"
#include <stdio.h>

#ifdef DEFINE_VARIABLES
#define EXTERN 
#else
#define EXTERN extern
#endif 
EXTERN texture<mbr_t> textoMBR;
EXTERN texture<mbr_t> textbMBR;


//================================ CreateMBR ===================================
__global__ void CreateMBR(mbr_t* bCoords, mbr_t* oCoords,long *bPolNum, long *oPolNum, int *bVNum, int *oVNum, int* bPrefixSum, int* oPrefixSum, mbr_t* bMBR, mbr_t* oMBR, mbr_t* allXMBR, mbr_t * allYMBR, int* xSortIndex, int* ySortIndex, int* xMBRIndex, int* yMBRIndex){
    __shared__ mbr_t MB[xThreadPerBlock][4], *sCoords, *sMBR;
    __shared__ int svPerThread, *svNum, *sprefixSum, spolBaseIndx;
    __shared__ long spolIndx, spolNum;    
        
    int tXIndx=threadIdx.x, polBaseIndx, vNum, vPerThread;
    mbr_t *Coords, *MBR;  
    long polIndx, gtIndx;

    gtIndx=blockIdx.y*gridDim.x+blockIdx.x;    
    
    if(gtIndx>=(*oPolNum + *bPolNum)){return;}

    if(tXIndx==0){
        spolIndx=-1;
        if(gtIndx<(*bPolNum)){                    
            sCoords=bCoords;
            spolNum=*bPolNum;
            svNum=bVNum;
            sMBR=bMBR;
            spolIndx=gtIndx;
            sprefixSum=bPrefixSum;     
        }
        else if(gtIndx<(*oPolNum + *bPolNum)){               
            sCoords=oCoords;
            spolNum=*oPolNum;
            svNum=oVNum;
            sMBR=oMBR;
            spolIndx=gtIndx-*bPolNum;
            sprefixSum=oPrefixSum;
        }  

        svPerThread=svNum[spolIndx]/xThreadPerBlock;
        if(svPerThread*xThreadPerBlock<svNum[spolIndx])svPerThread++;
        if(svPerThread==1)svPerThread=2;
  
        if(spolIndx==0){
            spolBaseIndx=0;
        }
        else if(spolIndx<spolNum && spolIndx!=-1){
            spolBaseIndx=sprefixSum[spolIndx-1]*2;
        }
    }
    
    __syncthreads();

    //polIndx=spolIndx;    
    MBR=sMBR;
    Coords=sCoords;
    vPerThread=svPerThread;        
    polBaseIndx=spolBaseIndx;
    
    if(spolIndx<spolNum && spolIndx!=-1){          
        vNum=svNum[spolIndx];  
        mbr_t x1=10000, x2=-10000, y1=10000, y2=-10000, x, y;
        for(int i=tXIndx*vPerThread;i<(tXIndx+1)*vPerThread;i++){
            if(i<vNum){
                x=Coords[polBaseIndx+2*i];
                y=Coords[polBaseIndx+i*2+1];
                if(x<x1)x1=x;
                if(y<y1)y1=y;
                if(x2<x)x2=x;
                if(y2<y)y2=y;
            }
            else break;
        }  
        MB[tXIndx][0]=x1;        
        MB[tXIndx][1]=y1;
        MB[tXIndx][2]=x2;
        MB[tXIndx][3]=y2;
    }    
    
    __syncthreads();
    
    int pw=2;
    while(xThreadPerBlock/pw>0){
        if(tXIndx%pw==0 && tXIndx+pw/2<xThreadPerBlock){
            if(MB[tXIndx+pw/2][0]<MB[tXIndx][0])MB[tXIndx][0]=MB[tXIndx+pw/2][0];
            if(MB[tXIndx+pw/2][1]<MB[tXIndx][1])MB[tXIndx][1]=MB[tXIndx+pw/2][1];
            if(MB[tXIndx+pw/2][2]>MB[tXIndx][2])MB[tXIndx][2]=MB[tXIndx+pw/2][2];
            if(MB[tXIndx+pw/2][3]>MB[tXIndx][3])MB[tXIndx][3]=MB[tXIndx+pw/2][3];
        }
        pw*=2;
        __syncthreads();
    }
    
    __syncthreads();
    
    if(tXIndx==0){  
        MBR[spolIndx*4]=MB[0][0];
        MBR[spolIndx*4+1]=MB[0][1];
        MBR[spolIndx*4+2]=MB[0][2];
        MBR[spolIndx*4+3]=MB[0][3];

	mbr_t roundingFactor=1000000, roundingOffset=100000000;
	allXMBR[gtIndx*2]=(roundingFactor*MB[0][0]+roundingOffset);
	allXMBR[gtIndx*2+1]=(roundingFactor*MB[0][2]+roundingOffset);
	allYMBR[gtIndx*2]=(roundingFactor*MB[0][1]+roundingOffset);
	allYMBR[gtIndx*2+1]=(roundingFactor*MB[0][3]+roundingOffset);
       
        xSortIndex[gtIndx*2]=gtIndx*2;
        xSortIndex[gtIndx*2+1]=gtIndx*2+1;
        ySortIndex[gtIndx*2]=gtIndx*2;
        ySortIndex[gtIndx*2+1]=gtIndx*2+1;

        xMBRIndex[gtIndx*2]=gtIndx*2;
        xMBRIndex[gtIndx*2+1]=gtIndx*2+1;
        yMBRIndex[gtIndx*2]=gtIndx*2;
        yMBRIndex[gtIndx*2+1]=gtIndx*2+1;
    } 
    return;
}
//==============================================================================


//=========================== GetOverlappedMBRs ==============================
__global__ void GetOverlappedMBRs(mbr_t *xMBR, mbr_t *yMBR, long *bPolNum, long* oPolNum, int* xSortIndex, int* ySortIndex, int* xMBRIndex, int* yMBRIndex, int *jxbCounter, int *jxbVector, int *jybCounter, int *jybVector, int *jxoCounter, int *jxoVector, int *jyoCounter, int *jyoVector){ 
    __shared__ int *jCounter[2], *jVector[2], *sortIndex[2], *mbrIndex[2], mbrRange1[2], mbrRange2[2], counter[2][512], PSCounter[2][512];

    int tXIndx=threadIdx.x, tYIndx=threadIdx.y, bXIndx=blockIdx.x, bYIndx=blockIdx.y, lIndx, uIndx, elementPerThread, itemPerThread;
    int elementIndx;
    elementIndx=bXIndx+bYIndx*gridDim.x;

    if(elementIndx >= *bPolNum+*oPolNum)return;

    if(tXIndx==0){
       if(tYIndx==0){
          sortIndex[0]=xSortIndex;
          sortIndex[1]=ySortIndex;
          mbrIndex[0]=xMBRIndex;
          mbrIndex[1]=yMBRIndex;
          if(elementIndx<*bPolNum){
	     mbrRange1[0]=0; 
	     mbrRange1[1]=2**bPolNum; 
	     mbrRange2[0]=2**bPolNum; 
	     mbrRange2[1]=2*(*bPolNum+*oPolNum); 
	     jVector[0]=jxbVector;
	     jVector[1]=jybVector;
	     jCounter[0]=jxbCounter;
	     jCounter[1]=jybCounter;
	   }
	   else{
	     mbrRange1[0]=2**bPolNum; 
	     mbrRange1[1]=2*(*bPolNum+*oPolNum); 
	     mbrRange2[0]=0; 
	     mbrRange2[1]=2**bPolNum; 
	     jVector[0]=jxoVector;
	     jVector[1]=jyoVector;
	     jCounter[0]=jxoCounter;
	     jCounter[1]=jyoCounter;
	   }
       } 
    }
   
    __syncthreads();

    lIndx=mbrIndex[tYIndx][elementIndx*2];
    uIndx=mbrIndex[tYIndx][elementIndx*2+1];
    itemPerThread=(uIndx-lIndx-1)/blockDim.x;
    //if(tXIndx==0 && tYIndx==0 && elementIndx==30547)printf("%d \t %d \t %d \t %d \t %d", elementIndx, lIndx, uIndx, blockDim.x, itemPerThread);
//return;

    if(itemPerThread*blockDim.x<uIndx-lIndx-1)itemPerThread++;
    int mIndx, cnt=0, shift, vector[1000];
    if(elementIndx>=*bPolNum){
       // Doing the overlay side 
       for(int i=lIndx+1+tXIndx*itemPerThread;i<lIndx+1+(tXIndx+1)*itemPerThread;i++){
         if(i>=uIndx)break;
         mIndx=sortIndex[tYIndx][i];
         if(mIndx/2>=*bPolNum)continue;
         if(mIndx%2==1)continue;
         if(mbrIndex[tYIndx][mIndx+1]>uIndx)continue; 

          vector[cnt++]=mIndx/2;

          if(cnt>1000){printf("O: %d :: %d :: itemPT:%d \t %d \t %d\n", elementIndx, tXIndx, uIndx-lIndx, itemPerThread, cnt);break;}
       }    
       counter[tYIndx][tXIndx]=cnt;
    }
    else{
       // Doing the base side
       for(int i=lIndx+1+tXIndx*itemPerThread;i<lIndx+1+(tXIndx+1)*itemPerThread;i++){
          if(i>=uIndx)break;
          mIndx=sortIndex[tYIndx][i];
          if(mIndx/2<*bPolNum)continue;
          if(mIndx%2==0 && mbrIndex[tYIndx][mIndx+1]<uIndx)continue;

          vector[cnt++]=mIndx/2-*bPolNum;

          if(cnt>1000){printf("B: %d :: %d :: itemPT:%d \t %d \t %d\n", elementIndx, tXIndx, uIndx-lIndx, itemPerThread, cnt);break;}
       }    
       counter[tYIndx][tXIndx]=cnt;
    }

    //if((elementIndx==*bPolNum || elementIndx==0) && cnt!=0)printf("%d:%d \t %d \t %d:(%d , %d) \t %d\n", tXIndx, tYIndx, elementIndx, itemPerThread, lIndx, uIndx, cnt);

    PSCounter[tYIndx][tXIndx]=counter[tYIndx][tXIndx];

    __syncthreads();

    int tempSum=0, p=1;
    while(p<blockDim.x){        
        if(tXIndx-p>=0){
            tempSum=PSCounter[tYIndx][tXIndx]+PSCounter[tYIndx][tXIndx-p];
        }                    
        __syncthreads();
        
        if(tXIndx-p>=0){
            PSCounter[tYIndx][tXIndx]=tempSum;
        }
        __syncthreads();
        
        p*=2;
    }

    __syncthreads();

    if(elementIndx>=*bPolNum){
       if(tXIndx==0)jCounter[tYIndx][(elementIndx-*bPolNum)]=PSCounter[tYIndx][511];
       shift=(elementIndx-*bPolNum)*GPU_MAX_JOIN_PER_DIM*2+2*PSCounter[tYIndx][tXIndx];
       for(int i=0;i<counter[tYIndx][tXIndx];i++){
         jVector[tYIndx][shift+2*i]=elementIndx-*bPolNum;
         jVector[tYIndx][shift+2*i+1]=vector[i];
      }
    }
    else{
       if(tXIndx==0)jCounter[tYIndx][elementIndx]=PSCounter[tYIndx][511];
       shift=elementIndx*GPU_MAX_JOIN_PER_DIM*2+2*(tXIndx>0?PSCounter[tYIndx][tXIndx-1]:0);
       for(int i=0;i<counter[tYIndx][tXIndx];i++){
         jVector[tYIndx][shift+2*i]=elementIndx;
         jVector[tYIndx][shift+2*i+1]=vector[i];
      }
    }

    return;
}
//==============================================================================


//=========================== GetOverlappedMBRs2 ==============================
__global__ void GetOverlappedMBRs2(mbr_t *xMBR, mbr_t *yMBR, long *bPolNum, long* oPolNum, int* xSortIndex, int* ySortIndex, int* xMBRIndex, int* yMBRIndex, int *jxoCounter, int *jxoVector, int *jyoCounter, int *jyoVector){ 
    __shared__ int *jCounter[2], *jVector[2], *sortIndex[2], *mbrIndex[2], counter[2][512], PSCounter[2][512];

    int tXIndx=threadIdx.x, tYIndx=threadIdx.y, bXIndx=blockIdx.x, bYIndx=blockIdx.y, lIndx, uIndx, elementPerThread, itemPerThread;
    int elementIndx;
    elementIndx=bXIndx+bYIndx*gridDim.x;

    if(elementIndx >= *oPolNum)return;

    if(tXIndx==0){
       if(tYIndx==0){
          sortIndex[0]=xSortIndex;
          sortIndex[1]=ySortIndex;
          mbrIndex[0]=xMBRIndex;
          mbrIndex[1]=yMBRIndex;
	     jVector[0]=jxoVector;
	     jVector[1]=jyoVector;
	     jCounter[0]=jxoCounter;
	     jCounter[1]=jyoCounter;
       } 
    }
   
    __syncthreads();

    lIndx=mbrIndex[tYIndx][elementIndx*2];
    uIndx=mbrIndex[tYIndx][elementIndx*2+1];
    itemPerThread=(uIndx-lIndx-1)/blockDim.x;
    //if(tXIndx==0 && tYIndx==0 && elementIndx==30547)printf("%d \t %d \t %d \t %d \t %d", elementIndx, lIndx, uIndx, blockDim.x, itemPerThread);
//return;

    if(itemPerThread*blockDim.x<uIndx-lIndx-1)itemPerThread++;
    int mIndx, cnt=0, shift=0, vector[1000];
       // Doing the overlay side 
       for(int i=lIndx+1+tXIndx*itemPerThread;i<lIndx+1+(tXIndx+1)*itemPerThread;i++){
         if(i>=uIndx)break;
         mIndx=sortIndex[tYIndx][i];
         if(mIndx/2>=*bPolNum)continue;
         if(mIndx%2==1)continue;
         if(mbrIndex[tYIndx][mIndx+1]>uIndx)continue; 

          vector[cnt++]=mIndx/2;

          if(cnt>1000){printf("O: %d :: %d :: itemPT:%d \t %d \t %d\n", elementIndx, tXIndx, uIndx-lIndx, itemPerThread, cnt);break;}
       }    
       counter[tYIndx][tXIndx]=cnt;

    //if((elementIndx==*bPolNum || elementIndx==0) && cnt!=0)printf("%d:%d \t %d \t %d:(%d , %d) \t %d\n", tXIndx, tYIndx, elementIndx, itemPerThread, lIndx, uIndx, cnt);

    PSCounter[tYIndx][tXIndx]=counter[tYIndx][tXIndx];

    __syncthreads();

    int tempSum=0, p=1;
    while(p<blockDim.x){        
        if(tXIndx-p>=0){
            tempSum=PSCounter[tYIndx][tXIndx]+PSCounter[tYIndx][tXIndx-p];
        }                    
        __syncthreads();
        
        if(tXIndx-p>=0){
            PSCounter[tYIndx][tXIndx]=tempSum;
        }
        __syncthreads();
        
        p*=2;
    }

    __syncthreads();

       if(tXIndx==0){
	  jCounter[tYIndx][elementIndx]=PSCounter[tYIndx][511];
          shift=elementIndx*GPU_MAX_JOIN_PER_DIM*2;
       }
       else{shift=elementIndx*GPU_MAX_JOIN_PER_DIM*2+2*PSCounter[tYIndx][tXIndx-1];}

       for(int i=0;i<counter[tYIndx][tXIndx];i++){
         jVector[tYIndx][shift+2*i]=elementIndx;
         jVector[tYIndx][shift+2*i+1]=vector[i];
      }
    return;
}
//==============================================================================

//=========================== CountSortBaseMBROverlapLoad ==============================
__global__ void CountSortBaseMBROverlapLoad(mbr_t *xMBR, mbr_t *yMBR, long bPolNum, long oPolNum, int* xSortIndex, int* ySortIndex, int* xMBRIndex, int* yMBRIndex, int *jxyCounter, char dimSort){ 
    __shared__ int PSCounter[1024];
    __shared__ int slIndx, suIndx, sItemPerThread, slIndx2, suIndx2;

    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y, lIndx, uIndx, elementPerThread, itemPerThread;
    int elementIndx;
    elementIndx=bXIndx+bYIndx*gridDim.x;
    if(elementIndx >= bPolNum+oPolNum)return;

    if(tXIndx==0){
       slIndx=xMBRIndex[elementIndx*2];
       suIndx=xMBRIndex[elementIndx*2+1];
       slIndx2=yMBRIndex[elementIndx*2];
       suIndx2=yMBRIndex[elementIndx*2+1];
       sItemPerThread=(suIndx-slIndx-1)/blockDim.x;
       if(sItemPerThread*blockDim.x<suIndx-slIndx-1)sItemPerThread++;
     } 
   
    __syncthreads();

    int mIndx, cnt=0;
    if(elementIndx>=bPolNum){
       // Doing the overlay side 
       for(int i=slIndx+1+tXIndx*sItemPerThread;i<slIndx+1+(tXIndx+1)*sItemPerThread;i++){
         if(i>=suIndx)break;
         mIndx=xSortIndex[i];
         if(mIndx>=2*bPolNum)continue;
         if(mIndx%2==1)continue;
	 //Check the other dimension
	 mbr_t y0, y1, b0, b1;
         y0=yMBR[2*elementIndx]; 
         b1=yMBR[mIndx+1]; 
	 if(y0>b1)continue;
         y1=yMBR[2*elementIndx+1]; 
         b0=yMBR[mIndx]; 
	 if(b0>y1)continue;
         cnt++;
       }    
    }
    else{
       // Doing the base side
       for(int i=slIndx+1+tXIndx*sItemPerThread;i<slIndx+1+(tXIndx+1)*sItemPerThread;i++){
          if(i>=suIndx)break;
          mIndx=xSortIndex[i];
          if( mIndx < 2*bPolNum)continue;
          if(mIndx%2==1)continue;
	 //Check the other dimension
	 mbr_t y0, y1, b0, b1;
         y0=yMBR[2*elementIndx]; 
         b1=yMBR[mIndx+1]; 
	 if(y0>b1)continue;
         y1=yMBR[2*elementIndx+1]; 
         b0=yMBR[mIndx]; 
	 if(b0>y1)continue;
         cnt++;
       }    
    }
    PSCounter[tXIndx]=cnt;

    __syncthreads();

    int tempSum=0, p=1;
    while(p<blockDim.x){        
        if(tXIndx-p>=0){
            tempSum=PSCounter[tXIndx]+PSCounter[tXIndx-p];
        }                    
        __syncthreads();
        if(tXIndx-p>=0){
            PSCounter[tXIndx]=tempSum;
        }
        __syncthreads();
        p*=2;
    }
    __syncthreads();
    if(tXIndx==0)jxyCounter[elementIndx]=PSCounter[blockDim.x-1];
    return;
}
//==============================================================================


//=========================== SortBaseMBROverlapPreloadCalculated ==============================
__global__ void SortBaseMBROverlapLoadCalculated(mbr_t *xMBR, mbr_t *yMBR, long bPolNum, long oPolNum, int* xSortIndex, int* ySortIndex, int* xMBRIndex, int* yMBRIndex, int *jxyCounter, int *jxyVector, long* jxyPSCounter, int dimSort){ 
    __shared__ int *sortIndex[2], *mbrIndex[2], counter[1024], PSCounter[1024], jShift;
    __shared__ int slIndx, suIndx, sItemPerThread, slIndx2, suIndx2;
    __shared__ mbr_t *otherSideMBR;

    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y, lIndx, uIndx, elementPerThread, itemPerThread;
    int elementIndx;

    elementIndx=bXIndx+bYIndx*gridDim.x;

    if(elementIndx >= bPolNum+oPolNum)return;

    if(tXIndx==0){
       if(elementIndx==0)jShift=0;
       else{
         jShift=2*(*(jxyPSCounter+elementIndx-1));
       }
       sortIndex[0]=xSortIndex;
       mbrIndex[0]=xMBRIndex;
       //sortIndex[1]=ySortIndex;
       mbrIndex[1]=yMBRIndex;
       otherSideMBR=yMBR;
       slIndx=mbrIndex[0][elementIndx*2];
       suIndx=mbrIndex[0][elementIndx*2+1];
       slIndx2=mbrIndex[1][elementIndx*2];
       suIndx2=mbrIndex[1][elementIndx*2+1];
       sItemPerThread=(suIndx-slIndx-1)/blockDim.x;
       if(sItemPerThread*blockDim.x<suIndx-slIndx-1)sItemPerThread++;
     } 
   
    __syncthreads();

    int mIndx, cnt=0, shift, vector[500];
    if(elementIndx>=bPolNum){
       // Doing the overlay side 
       for(int i=slIndx+1+tXIndx*sItemPerThread;i<slIndx+1+(tXIndx+1)*sItemPerThread;i++){
         if(i>=suIndx)break;
         mIndx=sortIndex[0][i];
         if(mIndx>=2*bPolNum)continue;
         if(mIndx%2==1)continue;
         //if(mbrIndex[0][mIndx+1]>uIndx)continue;

	 //Check the other dimension
	 //int mIndx0=(mIndx%2==0)?mIndx:mIndx-1;
	 mbr_t y0, y1, b0, b1;
         y0=otherSideMBR[2*elementIndx]; 
         b1=otherSideMBR[mIndx+1]; 
	 if(y0>b1)continue;
         y1=otherSideMBR[2*elementIndx+1]; 
         b0=otherSideMBR[mIndx]; 
	 if(b0>y1)continue;
	 //if(slIndx2>mbrIndex[1][mIndx+1])continue; 
	 //if(suIndx2<mbrIndex[1][mIndx])continue; 

          vector[cnt++]=mIndx/2;
       }    
       counter[tXIndx]=cnt;
    }
    else{
       // Doing the base side
       //if(elementIndx==35 && tXIndx==0)printf("\n%d \t %d\n", slIndx, suIndx);
       for(int i=slIndx+1+tXIndx*sItemPerThread;i<slIndx+1+(tXIndx+1)*sItemPerThread;i++){
          if(i>=suIndx)break;
          mIndx=*(sortIndex[0]+i);
          if( mIndx < 2*bPolNum)continue;
          if(mIndx%2==1)continue;
          //if(mbrIndex[0][mIndx+1]<uIndx)continue;

	 //Check the other dimension
	 //int mIndx0=(mIndx%2==0)?mIndx:mIndx-1;
	 mbr_t y0, y1, b0, b1;
         y0=otherSideMBR[2*elementIndx]; 
         b1=otherSideMBR[mIndx+1]; 
	 if(y0>b1)continue;
         y1=otherSideMBR[2*elementIndx+1]; 
         b0=otherSideMBR[mIndx]; 
	 if(b0>y1)continue;
	 //if(slIndx2>mbrIndex[1][mIndx+1])continue; 
  	 //if(suIndx2<mbrIndex[1][mIndx])continue; 
          vector[cnt++]=mIndx/2-bPolNum;
       }    
       counter[tXIndx]=cnt;
    }
    PSCounter[tXIndx]=counter[tXIndx];

    __syncthreads();

    int tempSum=0, p=1;
    while(p<blockDim.x){        
        if(tXIndx-p>=0){
            tempSum=PSCounter[tXIndx]+PSCounter[tXIndx-p];
        }                    
        __syncthreads();
        
        if(tXIndx-p>=0){
            PSCounter[tXIndx]=tempSum;
        }
        __syncthreads();
        
        p*=2;
    }

    __syncthreads();

       if(tXIndx==0)jxyCounter[elementIndx]=PSCounter[blockDim.x-1];
       shift=2*(tXIndx>0?PSCounter[tXIndx-1]:0);

       for(int i=0;i<counter[tXIndx];i++){
         if(elementIndx<bPolNum){
            jxyVector[jShift+shift+2*i]=elementIndx;
            jxyVector[jShift+shift+2*i+1]=vector[i];
	 }
	 else{
//if(elementIndx==193583)printf("\n%d\n", shift);
            jxyVector[jShift+shift+2*i]=vector[i];
            jxyVector[jShift+shift+2*i+1]=elementIndx-bPolNum;
	 }
      }
    return;
}
//==============================================================================


//=========================== CountOverlapingMBRs ==============================
__global__ void CountOverlapingMBRs(long bPolNum, long oPolNum, mbr_t* bMBR, mbr_t* oMBR, int* joinCounter, int * joinVector){    
    __shared__ int sJC[bBucketLength][oBucketLength], sJC_PFS[bBucketLength][oBucketLength];
    __shared__ mbr_t sbMBR[bBucketLength][4], soMBR[oBucketLength][4];
    __shared__ int sbIndx, soIndx, sbRepeat;    
    
    int tXIndx=threadIdx.x, tYIndx=threadIdx.y, bXIndx=blockIdx.x, bYIndx=blockIdx.y;        
    long blockIndx=bYIndx*gridDim.x+bXIndx;
    if(tXIndx==0 && tYIndx==0){
        sbRepeat=oPolNum/oBucketLength;
        if(sbRepeat*oBucketLength<oPolNum)sbRepeat++;        
        sbIndx=blockIndx/sbRepeat;
        soIndx=blockIndx%sbRepeat;
    }
    
    __syncthreads();  

    int bPolBaseIndx=(sbIndx*bBucketLength), oPolBaseIndx=(soIndx*oBucketLength);
    if(bPolBaseIndx>=bPolNum || oPolBaseIndx>=oPolNum) return;
    if(tYIndx==0 && tXIndx<bBucketLength && bPolBaseIndx+tXIndx<bPolNum){
        sbMBR[tXIndx][0]= bMBR[(bPolBaseIndx+tXIndx)*4];
        sbMBR[tXIndx][1]=bMBR[(bPolBaseIndx+tXIndx)*4+1];
        sbMBR[tXIndx][2]=bMBR[(bPolBaseIndx+tXIndx)*4+2];
        sbMBR[tXIndx][3]=bMBR[(bPolBaseIndx+tXIndx)*4+3];
    }
    else if(tYIndx==1 && tXIndx<oBucketLength && oPolBaseIndx+tXIndx<oPolNum){        
        soMBR[tXIndx][0]=oMBR[(oPolBaseIndx+tXIndx)*4];
        soMBR[tXIndx][1]=oMBR[(oPolBaseIndx+tXIndx)*4+1];
        soMBR[tXIndx][2]=oMBR[(oPolBaseIndx+tXIndx)*4+2];
        soMBR[tXIndx][3]=oMBR[(oPolBaseIndx+tXIndx)*4+3];
    }
    sJC[tXIndx][tYIndx]=0;   
    
    __syncthreads(); 

    mbr_t a1, b1, a2, b2, x1, x2, y1, y2;
    if(bPolBaseIndx+tXIndx<bPolNum && oPolBaseIndx+tYIndx<oPolNum){        
        a1=sbMBR[tXIndx][0];
        b1=sbMBR[tXIndx][1];
        a2=sbMBR[tXIndx][2];
        b2=sbMBR[tXIndx][3];
        x1=soMBR[tYIndx][0];
        if(a2>x1){
            y1=soMBR[tYIndx][1];
            if(b2>y1){
                x2=soMBR[tYIndx][2];
                if(x2>a1){
                    y2=soMBR[tYIndx][3];
                    if(y2>b1){
                        sJC[tXIndx][tYIndx]=1;                        
                    }
                }
            }
        }
    }
    
    sJC_PFS[tXIndx][tYIndx]=sJC[tXIndx][tYIndx];

    __syncthreads(); 
    
    int tempSum=0, p=1;
    while(p<blockDim.y){        
        if(tYIndx-p>=0){
                tempSum=sJC_PFS[tXIndx][tYIndx]+sJC_PFS[tXIndx][tYIndx-p];
        }                    
        __syncthreads();
        if(tYIndx-p>=0){
            sJC_PFS[tXIndx][tYIndx]=tempSum;
        }
        __syncthreads();
        p*=2;
    }    

    __syncthreads();
            
    int jBaseIndx=(sbIndx*bBucketLength+tXIndx)*sbRepeat+soIndx;    
    if(jBaseIndx>=bPolNum*sbRepeat)return;   


    if(tYIndx==0){
        *(joinCounter+jBaseIndx)=sJC_PFS[tXIndx][blockDim.y-1];
        if(sJC_PFS[tXIndx][blockDim.y-1]>GPU_MAX_CROSS_JOIN)printf("\n%ld block jVector overflow (%d required, %d available)\n", blockIndx, sJC_PFS[tXIndx][blockDim.y-1], GPU_MAX_CROSS_JOIN);
    }    

    __syncthreads();

    int shift=0;
    if(sJC[tXIndx][tYIndx]>0){        
        if(tYIndx>0)shift=sJC_PFS[tXIndx][tYIndx-1];
        joinVector[jBaseIndx*GPU_MAX_CROSS_JOIN+shift]=oPolBaseIndx+tYIndx;
    }        
    return;
}
//==============================================================================

//=========================== CountOverlapingMBRs2 =============================
__global__ void CountOverlapingMBRs2(long *bPolNum, long *oPolNum, mbr_t* bMBR, mbr_t* oMBR, int* joinCounter, int * joinVector){    
    __shared__ int sJC[1024], sJC_PFS[1024];
    __shared__ mbr_t a1, b1, a2, b2;
    __shared__ int soTestPerThread;
        
    int JV[130];
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y;        
    long bIndx=bYIndx*gridDim.x+bXIndx;
    
    if(bIndx>=*bPolNum)return;
    
    mbr_t  x1, x2, y1, y2;
    if(tXIndx==0){
        /*a1=tex1Dfetch(textbMBR, (bIndx)*4);
        b1=tex1Dfetch(textbMBR, (bIndx)*4+1);
        a2=tex1Dfetch(textbMBR, (bIndx)*4+2);
        b2=tex1Dfetch(textbMBR, (bIndx)*4+3); */ 
        soTestPerThread=*oPolNum/1024;
    }
    
    __syncthreads();  
    
    sJC[tXIndx]=0;
    for(int i=tXIndx*soTestPerThread;i<(tXIndx+1)*soTestPerThread;i++){   
        if(i>=*oPolNum)break;
       // x1=tex1Dfetch(textoMBR, (i)*4);        
        if(a2>x1){
            //y1=tex1Dfetch(textoMBR, (i)*4+1);
            if(b2>y1){
                //x2=tex1Dfetch(textoMBR, (i)*4+2);
                if(x2>a1){
                    //y2=tex1Dfetch(textoMBR, (i)*4+3);
                    if(y2>b1){
                        JV[sJC[tXIndx]++]=i;                        
                    }
                }
            }
        }
    }
    __syncthreads(); 
    
    sJC_PFS[tXIndx]=sJC[tXIndx];
    
    __syncthreads();
    
    int tempSum=0, p=1;
    while(p<blockDim.x){        
        if(tXIndx-p>=0){
                tempSum=sJC_PFS[tXIndx]+sJC_PFS[tXIndx-p];
        }                    
        __syncthreads();
        if(tXIndx-p>=0){
            sJC_PFS[tXIndx]=tempSum;
        }
        __syncthreads();
        p*=2;
    }    

    __syncthreads();                    
    
    if(tXIndx==0){        
        *(joinCounter+bIndx)=sJC_PFS[blockDim.x-1];
    }    
    int shift=0;
    if(sJC[tXIndx]>0){        
        if(tXIndx>0)shift=sJC_PFS[tXIndx-1];
        for(int i=0;i<sJC[tXIndx];i++){
            //joinVector[2*(bIndx*GPU_MAX_CROSS_JOIN_PER_BASE+shift+i)]=bIndx;            
            joinVector[(bIndx*GPU_MAX_CROSS_JOIN_PER_BASE+shift+i)]=JV[i];
        }
    }        
    return;
}
//==============================================================================

//=========================== CalculateMBRLoad =============================
__global__ void CalculateMBRLoad(int *xMBRIndex, int* yMBRIndex, int* xMBRLoadCounter, int* yMBRLoadCounter, int mbrNum, int dimSort){
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x;        
    int elementIndx= bXIndx*blockDim.x+tXIndx;
    if(elementIndx>=mbrNum)return;

    *(xMBRLoadCounter+elementIndx)=(*(xMBRIndex+2*elementIndx+1)-*(xMBRIndex+2*elementIndx))*INTERSECT_RATE;
    //if(*(xMBRLoadCounter+elementIndx)<0)printf("\nNegative at : %d\n", elementIndx);
    if(dimSort==2)*(yMBRLoadCounter+elementIndx)=(*(yMBRIndex+2*elementIndx+1)-*(yMBRIndex+2*elementIndx))*INTERSECT_RATE;

   return;
}

//==============================================================================


//=========================== GetOverlappedMBRs3 ==============================
__global__ void GetOverlappedMBRs4(long bPolNum, long oPolNum, int* xSortIndex, int* ySortIndex, int* xMBRIndex, int* yMBRIndex, int *xMBRLoadPSCounter, int* yMBRLoadPSCounter, int *jxyCounter, int *jxyVector, int dimSort){ 
    __shared__ int *sortIndex[2], *mbrIndex1, *mbrIndex2, counter[1024], PSCounter[1024], jShift, maxISect;
    __shared__ int slIndx, suIndx, sItemPerThread, slIndx2, suIndx2;
    __shared__ mbr_t *otherSideMBR;

    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y, lIndx, uIndx, elementPerThread, itemPerThread, sItemPerThread2;
    int elementIndx;

    elementIndx=bXIndx+bYIndx*gridDim.x;

    if(elementIndx >= bPolNum+oPolNum)return;

    if(tXIndx==0){
       if(dimSort==1){
	  sortIndex[0]=xSortIndex;
       	  mbrIndex1=xMBRIndex;
       	  //sortIndex[1]=ySortIndex;
       	  mbrIndex2=yMBRIndex;
       }
       else{
          if(xMBRLoadPSCounter[bPolNum+oPolNum-1]<yMBRLoadPSCounter[bPolNum+oPolNum-1]){
             maxISect=xMBRLoadPSCounter[elementIndx];
             if(elementIndx!=0)maxISect-=xMBRLoadPSCounter[elementIndx-1];
             if(elementIndx==0)jShift=0;
             else jShift=2*(*(xMBRLoadPSCounter+elementIndx-1)+(elementIndx-1)*INTERSECT_CONST);
	     sortIndex[0]=ySortIndex;
       	     mbrIndex1=yMBRIndex;
       	     //sortIndex[1]=xSortIndex;
       	     mbrIndex2=xMBRIndex;
          }
          else{
             maxISect=yMBRLoadPSCounter[elementIndx];
             if(elementIndx!=0)maxISect-=yMBRLoadPSCounter[elementIndx-1];
             if(elementIndx==0)jShift=0;
             else jShift=2*(*(yMBRLoadPSCounter+elementIndx-1)+(elementIndx-1)*INTERSECT_CONST);
	     sortIndex[0]=xSortIndex;
       	     mbrIndex1=xMBRIndex;
       	     //sortIndex[1]=ySortIndex;
       	     mbrIndex2=yMBRIndex;
          }
       }

       slIndx=mbrIndex1[elementIndx*2];
       suIndx=mbrIndex1[elementIndx*2+1];
       slIndx2=mbrIndex2[elementIndx*2];
       suIndx2=mbrIndex2[elementIndx*2+1];
       sItemPerThread=(suIndx-slIndx-1)/blockDim.x;
       sItemPerThread2=(suIndx2-slIndx2-1)/blockDim.x;
       if(sItemPerThread*blockDim.x<suIndx-slIndx-1)sItemPerThread++;
     } 
   
    __syncthreads();

    int mIndx, cnt=0, shift, vector[500];
    if(elementIndx>=bPolNum){
       // Doing the overlay side 
       for(int i=slIndx+1+tXIndx*sItemPerThread;i<slIndx+1+(tXIndx+1)*sItemPerThread;i++){
         if(i>=suIndx)break;
         mIndx=sortIndex[0][i];
         if(mIndx>=2*bPolNum)continue;
         if(mIndx%2==1)continue;
       }

       for(int i=slIndx2+1+tXIndx*sItemPerThread2;i<slIndx2+1+(tXIndx+1)*sItemPerThread2;i++){
	 if(slIndx2>mbrIndex2[mIndx+1])continue; 
         vector[cnt++]=mIndx/2-bPolNum;
         if(cnt>GPU_MAX_CROSS_JOIN_PER_BASE){printf("Running out of memory for saving results:B: %d::%d :: itemPT:%d \t %d \t %d\n",elementIndx,tXIndx,uIndx-lIndx,itemPerThread,cnt);break;}
	 if(slIndx2>mbrIndex2[mIndx+1])continue; 
	 if(suIndx2<mbrIndex2[mIndx])continue; 
         vector[cnt++]=mIndx/2;
         if(cnt>GPU_MAX_CROSS_JOIN_PER_BASE){printf("Running out of memory for saving results:O: %d::%d::itemPT:%d \t %d \t %d\n", elementIndx, tXIndx, uIndx-lIndx, itemPerThread, cnt);break;}
       }    

       counter[tXIndx]=cnt;
    }
    else{
       // Doing the base side
       for(int i=slIndx+1+tXIndx*sItemPerThread;i<slIndx+1+(tXIndx+1)*sItemPerThread;i++){
          if(i>=suIndx)break;
          mIndx=*(sortIndex[0]+i);
          if( mIndx < 2*bPolNum)continue;
          if(mIndx%2==1)continue;
        }

       for(int i=slIndx2+1+tXIndx*sItemPerThread2;i<slIndx2+1+(tXIndx+1)*sItemPerThread2;i++){
	 if(slIndx2>mbrIndex2[mIndx+1])continue; 
          vector[cnt++]=mIndx/2-bPolNum;
          if(cnt>GPU_MAX_CROSS_JOIN_PER_BASE){printf("Running out of memory for saving results:B: %d::%d :: itemPT:%d \t %d \t %d\n",elementIndx,tXIndx,uIndx-lIndx,itemPerThread,cnt);break;}
       }    

       counter[tXIndx]=cnt;
    }
    PSCounter[tXIndx]=counter[tXIndx];

return;
    __syncthreads();
    int tempSum=0, p=1;
    while(p<blockDim.x){        
        if(tXIndx-p>=0){
            tempSum=PSCounter[tXIndx]+PSCounter[tXIndx-p];
        }                    
        __syncthreads();
        if(tXIndx-p>=0){
            PSCounter[tXIndx]=tempSum;
        }
        __syncthreads();
        p*=2;
    }

    __syncthreads();

       if(tXIndx==0)jxyCounter[elementIndx]=PSCounter[blockDim.x-1];
       //shift=elementIndx*GPU_MAX_JOIN_PER_DIM*2+2*(tXIndx>0?PSCounter[tXIndx-1]:0);
       shift=2*(tXIndx>0?PSCounter[tXIndx-1]:0);
       for(int i=0;i<counter[tXIndx];i++){
         if(elementIndx<bPolNum){
            jxyVector[jShift+shift+2*i]=elementIndx;
            jxyVector[jShift+shift+2*i+1]=vector[i];
	 }
	 else{
            jxyVector[jShift+shift+2*i]=vector[i];
            jxyVector[jShift+shift+2*i+1]=elementIndx-bPolNum;
	 }
      }
    return;
}
//==============================================================================
