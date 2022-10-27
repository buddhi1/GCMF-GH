#include "Constants.h"
#include "Types.h"
#include <stdio.h>


//============================== IntersectSide =================================
__global__ void IntersectSideEbE(mbr_t *bCoords, mbr_t * oCoords, long *bPolNum, long* oPolNum, long *pairNum, int* PFSideCrossCounter, int* jCompactVector, int* bVNum, int* oVNum, int *bPFVNum, int* oPFVNum, mbr_t* sideIntersectionVector){    
    int matchedSideCrossIndx=-1;
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y, oIndx, bIndx;
    int blockIndx=bYIndx*gridDim.x+bXIndx, gtIndx=blockIndx*blockDim.x+tXIndx;
    
    long crossSideNum=PFSideCrossCounter[*pairNum-1];
    
    if(gtIndx>=crossSideNum)return;
    
    long l1_Indx=0, l2_Indx=*pairNum-1, mIndx=(l1_Indx+l2_Indx)/2;
    while(1){             
        if(l1_Indx>=crossSideNum)return;
        if(gtIndx+1<PFSideCrossCounter[mIndx] && mIndx==0){matchedSideCrossIndx=0;break;}
        if(PFSideCrossCounter[mIndx-1]<gtIndx+1 && gtIndx+1<=PFSideCrossCounter[mIndx]){matchedSideCrossIndx=mIndx;break;}
        if(PFSideCrossCounter[mIndx-1]>=gtIndx+1)l2_Indx=mIndx;
        if(PFSideCrossCounter[mIndx]<gtIndx+1)l1_Indx=mIndx;
        mIndx=(l1_Indx+l2_Indx)/2;        
    }
    oIndx=jCompactVector[2*matchedSideCrossIndx+1];
    bIndx=jCompactVector[2*matchedSideCrossIndx];

    
    int sideCrossNumForThisPol, oVN;
    oVN=oVNum[oIndx];

    sideCrossNumForThisPol=gtIndx;
    if(matchedSideCrossIndx!=0)sideCrossNumForThisPol-=PFSideCrossCounter[matchedSideCrossIndx-1];
    
    int bCoordIndx, oCoordIndx, bBase=0, oBase=0;
    if(bIndx!=0)bBase=bPFVNum[bIndx-1];
    if(oIndx!=0)oBase=oPFVNum[oIndx-1];    
    bCoordIndx=(sideCrossNumForThisPol)/oVN;
    oCoordIndx=(sideCrossNumForThisPol)%oVN;

    mbr_t bX1, bX2, bY1, bY2, oX1, oX2, oY1, oY2;
    bX1=bCoords[2*(bBase+bCoordIndx)];
    bY1=bCoords[2*(bBase+bCoordIndx)+1];
    bX2=bCoords[2*(bBase+bCoordIndx+1)];
    bY2=bCoords[2*(bBase+bCoordIndx+1)+1];
    oX1=oCoords[2*(oBase+oCoordIndx)];
    oY1=oCoords[2*(oBase+oCoordIndx)+1];
    oX2=oCoords[2*(oBase+oCoordIndx+1)];
    oY2=oCoords[2*(oBase+oCoordIndx+1)+1];
    
    float m1, m2, x0, y0;
    m1=(bY2-bY1)/(bX2-bX1);
    m2=(oY2-oY1)/(oX2-oX1);
    x0=((oY1-bY1)+m1*bX1-m2*oX1)/(m1-m2);
    y0=m1*(x0-bX1)+bY1;
    if( (bX1<=x0) && (x0<=bX2) && (oX1<=x0) && (x0<=oX2) && (bY1<=y0) && (y0<=bY2) && (oY1<=y0) && (y0<=oY2) ){        
        //sideIntersectionVector[2*gtIndx]=x0;
        //sideIntersectionVector[2*gtIndx+1]=y0;
        //if(x0==-1&&y0==-1)printf("(-1 , -1) is happened!");return;
    }
    else{        
        //sideIntersectionVector[2*gtIndx]=-1;
        //sideIntersectionVector[2*gtIndx+1]=-1;
    }
    return;
    }
//==============================================================================

//=============================== SwapElements =================================
__device__ void SwapElements(double *SideIntersectionVector, int *directPointer, int* reversePointer, int i, int j, int blockShift){    
    double dTmp;
    dTmp=SideIntersectionVector[i];
    SideIntersectionVector[i]=SideIntersectionVector[j];
    SideIntersectionVector[j]=dTmp;
    
    int oIndx_i=directPointer[i]+blockShift, oIndx_j=directPointer[j]+blockShift;
    int iTmp=reversePointer[oIndx_i];
    reversePointer[oIndx_i]=reversePointer[oIndx_j];
    reversePointer[oIndx_j]=iTmp;

    iTmp=directPointer[i];
    directPointer[i]=directPointer[j];
    directPointer[j]=iTmp;                        
}
//==============================================================================
                    
//============================= IntersectSidePpB ===============================
__global__ void IntersectSidePpB(mbr_t *bCoords, mbr_t * oCoords, long *bPolNum, long* oPolNum, int* jPSCounter, int* jCompactVector, int* bVNum, int* oVNum, int *bPFVNum, int* oPFVNum, double* bSideIntersectionVector, double* oSideIntersectionVector, int* sideCrossCounter, int* PFSideCrossCounter, int* b2oPointer, int* o2bPointer){
    __shared__ int oIndx, bIndx, crossSidePerThread, oV, bV, bBase, oBase, SICounter[1024], SICounter_PFS[1024], sortedFlag, sblockShift;    
    
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y;
    int blockIndx=bYIndx*gridDim.x+bXIndx, gtIndx=blockIndx*blockDim.x+tXIndx, pairNum=jPSCounter[*bPolNum-1];                
    
    if(blockIndx>=pairNum)return;
    if(tXIndx==0){
	sblockShift=0;
	if(blockIndx>0)sblockShift=PFSideCrossCounter[blockIndx-1];
        bBase=0;
        oBase=0;
        bIndx=jCompactVector[2*(blockIndx)];        
        oIndx=jCompactVector[2*(blockIndx)+1];        
        oV=oVNum[oIndx];
        bV=bVNum[bIndx];
        crossSidePerThread=bV*oV/blockDim.x;
        if(crossSidePerThread<1)crossSidePerThread=1;
        if(bIndx!=0)bBase=bPFVNum[bIndx-1];
        if(oIndx!=0)oBase=oPFVNum[oIndx-1];    
    }
    double bSIVector[MAX_EI_PER_THREAD], oSIVector[MAX_EI_PER_THREAD];

    __syncthreads();   
    
    SICounter[tXIndx]=0;
    if(sideCrossCounter[blockIndx]==0){
     //Point in Polygon Test
	return;
    }
    else{        

	int bCoordIndx, oCoordIndx;
        int crossSideCounter=0, flag=0;
        mbr_t bX1, bX2, bY1, bY2, oX1, oX2, oY1, oY2;
        float m1, m2, x0, y0;        
    	if(tXIndx*crossSidePerThread<oV*bV){
	    bCoordIndx=tXIndx*crossSidePerThread/oV;
	    oCoordIndx=tXIndx*crossSidePerThread%oV;
	    int counter=0;
	    mbr_t bX1, bX2, bY1, bY2, oX1, oX2, oY1, oY2;
	    float m1, m2, x0, y0;
	    for(int i=bCoordIndx;i<bV && counter<crossSidePerThread;i++){
		int j;
		j=0;
		if(i==bCoordIndx)j=oCoordIndx;        
		bX1=bCoords[2*(bBase+i)];
		bY1=bCoords[2*(bBase+i)+1];
		if(i<bV-1){
			bX2=bCoords[2*(bBase+i+1)];
			bY2=bCoords[2*(bBase+i+1)+1];
		}
		else{
			bX2=bCoords[2*(bBase+0)];
			bY2=bCoords[2*(bBase+0)+1];
		}
		for(;j<oV && counter<crossSidePerThread;j++){
		    counter++;
		    oX1=oCoords[2*(oBase+j)];
		    oY1=oCoords[2*(oBase+j)+1];
		    if(j<oV-1){
			oX2=oCoords[2*(oBase+j+1)];
			oY2=oCoords[2*(oBase+j+1)+1];
		    }
		    else{
			oX2=oCoords[2*(oBase+0)];
			oY2=oCoords[2*(oBase+0)+1];
		    }
		    m1=(bY2-bY1)/(bX2-bX1);
		    m2=(oY2-oY1)/(oX2-oX1);
		    x0=((oY1-bY1)+m1*bX1-m2*oX1)/(m1-m2);
		    y0=m1*(x0-bX1)+bY1;
		    if( (bX1<=x0) && (x0<=bX2) && (oX1<=x0) && (x0<=oX2) && (bY1<=y0) && (y0<=bY2) && (oY1<=y0) && (y0<=oY2) ){
                    	if(SICounter[tXIndx]<MAX_EI_PER_THREAD){
                        	double d_b, d_o;
                        	double bEdgeLen, oEdgeLen;
				bEdgeLen=((bX2-bX1)*(bX2-bX1)+(bY2-bY1)*(bY2-bY1))*1.01+0.0001;
				oEdgeLen=((oX2-oX1)*(oX2-oX1)+(oY2-oY1)*(oY2-oY1))*1.01+0.0001;
				d_b=(((x0-bX1)*(x0-bX1)+(y0-bY1)*(y0-bY1))*1.005+0.000000001)/bEdgeLen;
				d_o=(((x0-oX1)*(x0-oX1)+(y0-oY1)*(y0-oY1))*1.005+0.000000001)/oEdgeLen;
				
				/*bEdgeLen=((bX2-bX1)*(bX2-bX1)+(bY2-bY1)*(bY2-bY1));
				oEdgeLen=((oX2-oX1)*(oX2-oX1)+(oY2-oY1)*(oY2-oY1));
				d_b=(((x0-bX1)*(x0-bX1)+(y0-bY1)*(y0-bY1)))/bEdgeLen;
				d_o=(((x0-oX1)*(x0-oX1)+(y0-oY1)*(y0-oY1)))/oEdgeLen;*/

				bSIVector[SICounter[tXIndx]]=10*i+d_b;
				oSIVector[SICounter[tXIndx]]=10*j+d_o;
				SICounter[tXIndx]++;
                    	}
                    	else{
                        	printf("\nmore than enough space per thread in %d of %d with %d (%d,%d)!\n", tXIndx, blockIndx, crossSidePerThread, bV, oV);
                        	return;
                    	}
		    }            
		}
	    }
    }
   

        SICounter_PFS[tXIndx]=SICounter[tXIndx];

        __syncthreads();
    
        int tempSum=0, p=1;
        while(p<blockDim.x){        
            if(tXIndx-p>=0){
                tempSum=SICounter_PFS[tXIndx]+SICounter_PFS[tXIndx-p];
            }                    
            __syncthreads();

            if(tXIndx-p>=0){
                SICounter_PFS[tXIndx]=tempSum;
            }
            __syncthreads();

            p*=2;
        }
       
	__syncthreads();
 
        if(SICounter[tXIndx]!=0){
            int shift=0;
            if(tXIndx>0)shift=SICounter_PFS[tXIndx-1];
            for(int i=0;i<SICounter[tXIndx];i++){
                bSideIntersectionVector[sblockShift+shift+i]=bSIVector[i];
                oSideIntersectionVector[sblockShift+shift+i]=oSIVector[i];
                b2oPointer[sblockShift+shift+i]=shift+i;
                o2bPointer[sblockShift+shift+i]=shift+i;
            }
        }
    
             __syncthreads();
   //return;                   


            int elementNum=SICounter_PFS[1023];
            int elementPerThread=elementNum/blockDim.x;
	    if(elementPerThread<2)elementPerThread=2;
            int startIndx;

  /* if(blockIndx==11794 && tXIndx==0){
      for(int i=sblockShift;(i-sblockShift)<elementNum;i++){
	 printf("\n%f , %d , %f , %d\n", bSideIntersectionVector[i],b2oPointer[i], oSideIntersectionVector[i],o2bPointer[i]);		
	}
    }
    __syncthreads();*/

            while(1){
                sortedFlag=0;
                startIndx=tXIndx*elementPerThread;
                if(startIndx%2 != 0)startIndx++;
                for(int i=startIndx+sblockShift;(i-sblockShift)<(tXIndx+1)*elementPerThread && (i-sblockShift)+1<elementNum;i+=2){

		    //if(blockIndx==11794)printf("\n\n\nb: %f----%d:::%d---> %f\n\n\n",bSideIntersectionVector[i],i,elementNum,bSideIntersectionVector[i+1]);

                    if(bSideIntersectionVector[i]>bSideIntersectionVector[i+1]){
                        sortedFlag++;
                        SwapElements(bSideIntersectionVector, b2oPointer, o2bPointer, i, i+1, sblockShift);
                    }
                }
                
                __syncthreads();
                
                startIndx=tXIndx*elementPerThread;
                if(startIndx%2 != 1)startIndx++;
                for(int i=sblockShift+startIndx;(i-sblockShift)<(tXIndx+1)*elementPerThread;i+=2){

		   // if(blockIndx==11794)printf("\n\n\nb: %f----%d:::%d---> %f\n\n\n",bSideIntersectionVector[i],i,elementNum,bSideIntersectionVector[i+1]);

                    if((i-sblockShift)+1<elementNum && bSideIntersectionVector[i]>bSideIntersectionVector[i+1]){
                        sortedFlag++;
                        SwapElements(bSideIntersectionVector, b2oPointer, o2bPointer, i, i+1, sblockShift);
                    }
                }

		__syncthreads();

                if(sortedFlag==0)break;
            }

            while(1){
                sortedFlag=0;
                startIndx=tXIndx*elementPerThread;
                if(startIndx%2 != 0)startIndx++;
                for(int i=startIndx+sblockShift;(i-sblockShift)<(tXIndx+1)*elementPerThread && (i-sblockShift)+1<elementNum;i+=2){

		    //if(blockIndx==11794)printf("\n\n\nb: %f----%d:::%d---> %f\n\n\n",oSideIntersectionVector[i],i,elementNum,oSideIntersectionVector[i+1]);

                    if(oSideIntersectionVector[i]>oSideIntersectionVector[i+1]){
                        sortedFlag++;
                        SwapElements(oSideIntersectionVector, o2bPointer, b2oPointer, i, i+1, sblockShift);
                    }
                }
                
                __syncthreads();
                
                startIndx=tXIndx*elementPerThread;
                if(startIndx%2 != 1)startIndx++;
                for(int i=sblockShift+startIndx;(i-sblockShift)<(tXIndx+1)*elementPerThread;i+=2){

		   // if(blockIndx==11794)printf("\n\n\nb: %f----%d:::%d---> %f\n\n\n",oSideIntersectionVector[i],i,elementNum,oSideIntersectionVector[i+1]);

                    if((i-sblockShift)+1<elementNum && oSideIntersectionVector[i]>oSideIntersectionVector[i+1]){
                        sortedFlag++;
                        SwapElements(oSideIntersectionVector, o2bPointer, b2oPointer, i, i+1, sblockShift);
                    }
                }

		__syncthreads();

                if(sortedFlag==0)break;   
            }
 
    }
    return;
    }
 //==============================================================================


//=========================== IntersectSideCounter =============================
__global__ void IntersectSideCounter(mbr_t *bCoords, mbr_t * oCoords, long *bPolNum, long* oPolNum, int* jPSCounter, int* jCompactVector, int* bVNum, int* oVNum, int *bPFVNum, int* oPFVNum, int* sideCrossCounter){
    __shared__ int oIndx, bIndx, crossSidePerThread, oV, bV, bBase, oBase, crossCounter_PFS[1024];
    
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y, crossCounter;
    int blockIndx=bYIndx*gridDim.x+bXIndx, gtIndx=blockIndx*blockDim.x+tXIndx, pairNum;        
    
    crossCounter=0;
    
    pairNum=jPSCounter[*bPolNum-1];

    if(blockIndx>=pairNum)return;
    if(tXIndx==0){
        bBase=0;
        oBase=0;
        bIndx=jCompactVector[2*(blockIndx)];        
        oIndx=jCompactVector[2*(blockIndx)+1];        
        oV=oVNum[oIndx];
        bV=bVNum[bIndx];
        crossSidePerThread=bV*oV/blockDim.x;
        if(crossSidePerThread<1)crossSidePerThread=1;
        if(bIndx!=0)bBase=bPFVNum[bIndx-1];
        if(oIndx!=0)oBase=oPFVNum[oIndx-1];    
    }

    __syncthreads();   
    
    int bCoordIndx, oCoordIndx;
    if(tXIndx*crossSidePerThread<oV*bV){
	    bCoordIndx=tXIndx*crossSidePerThread/oV;
	    oCoordIndx=tXIndx*crossSidePerThread%oV;
	    int counter=0;
	    mbr_t bX1, bX2, bY1, bY2, oX1, oX2, oY1, oY2;
	    float m1, m2, x0, y0;
	    for(int i=bCoordIndx;i<bV && counter<crossSidePerThread;i++){
		int j;
		j=0;
		if(i==bCoordIndx)j=oCoordIndx;        
		bX1=bCoords[2*(bBase+i)];
		bY1=bCoords[2*(bBase+i)+1];
		if(i<bV-1){
			bX2=bCoords[2*(bBase+i+1)];
			bY2=bCoords[2*(bBase+i+1)+1];
		}
		else{
			bX2=bCoords[2*(bBase+0)];
			bY2=bCoords[2*(bBase+0)+1];
		}
		for(;j<oV && counter<crossSidePerThread;j++){
		    counter++;
		    oX1=oCoords[2*(oBase+j)];
		    oY1=oCoords[2*(oBase+j)+1];
		    if(j<oV-1){
			oX2=oCoords[2*(oBase+j+1)];
			oY2=oCoords[2*(oBase+j+1)+1];
		    }
		    else{
			oX2=oCoords[2*(oBase+0)];
			oY2=oCoords[2*(oBase+0)+1];
		    }
		    m1=(bY2-bY1)/(bX2-bX1);
		    m2=(oY2-oY1)/(oX2-oX1);
		    x0=((oY1-bY1)+m1*bX1-m2*oX1)/(m1-m2);
		    y0=m1*(x0-bX1)+bY1;
		    if( (bX1<=x0) && (x0<=bX2) && (oX1<=x0) && (x0<=oX2) && (bY1<=y0) && (y0<=bY2) && (oY1<=y0) && (y0<=oY2) ){
			crossCounter++;
		    }            
		}
	    }
    }
    crossCounter_PFS[tXIndx]=crossCounter;

    __syncthreads();
    
    int tempSum=0, p=1;
    while(p<blockDim.x){        
        if(tXIndx-p>=0){
            tempSum=crossCounter_PFS[tXIndx]+crossCounter_PFS[tXIndx-p];
        }                    
        __syncthreads();
        
        if(tXIndx-p>=0){
            crossCounter_PFS[tXIndx]=tempSum;
        }
        __syncthreads();
        
        p*=2;
    }

    if(tXIndx==0)sideCrossCounter[blockIndx]=crossCounter_PFS[1023];
    return;
    }
//==============================================================================


//============================ PointInPolygonTest ==============================
__global__ void PointInPolygonTest(mbr_t *Coords, int num){

    return;
}
//==============================================================================


//============================== UpdateEntryExit ===============================
void GetEdgeIndex(double *crossSideVector, int indx, int* iEdgeIndx){
    long int temp=crossSideVector[indx];
    temp/=10;
    *iEdgeIndx=temp;
    return;
}

//==============================================================================



//============================== UpdateEntryExit ===============================
__global__ void UpdateEntryExit(mbr_t* bCoords, mbr_t* oCoords, long* bPolNum, long* oPolNum, int* bVNum, int* oVNum, int* bVNumPSum, int* oVNumPSum, int* jPSCounter, int* jCompactVector, double* bCrossSideVector, double* oCrossSideVector, int* SideCrossCounter, int* PFSideCrossCounter){
    __shared__ int sVPerThread, oIndx, bIndx, sBlockShift, sVNum, sShift, sEIShift, sInitEEVal, sEINum;
    __shared__ double *sCrossSideVector;
    __shared__ mbr_t sX0, sY0, *sCoords;
    __shared__ bool sStatus[1024];
    
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y;
    int blockIndx=bYIndx*gridDim.x+bXIndx, gtIndx=blockIndx*blockDim.x+tXIndx, pairNum=jPSCounter[*bPolNum-1];                
    int bCoordIndx, oCoordIndx, pairIndx;
    
    if(blockIndx>=2*pairNum)return;
    pairIndx=blockIndx/2;
    
    if(tXIndx==0){
        bIndx=jCompactVector[2*(pairIndx)];        
        oIndx=jCompactVector[2*(pairIndx)+1];                
        sEINum=SideCrossCounter[pairIndx];
        if(blockIndx%2==0){
            sCrossSideVector=bCrossSideVector;
            sCoords=bCoords;
            sShift=0;
            if(bIndx>0)sShift=bVNumPSum[bIndx-1];            
            sVNum=bVNum[bIndx];            
            int shift=0;
            if(oIndx>0)shift=oVNumPSum[oIndx-1];                    
            sX0=oCoords[2*shift];
            sY0=oCoords[2*shift+1];            
        }   
        else{
            sCrossSideVector=oCrossSideVector;
            sCoords=oCoords;
            sShift=0;
            if(oIndx>0)sShift=oVNumPSum[oIndx-1];
            sVNum=oVNum[oIndx];
            int shift=0;
            if(bIndx>0)shift=bVNumPSum[bIndx-1];                    
            sX0=bCoords[2*shift];
            sY0=bCoords[2*shift+1];            
        }
        sVPerThread=sVNum/blockDim.x;
        sEIShift=0;
        if(pairIndx>0)sEIShift=PFSideCrossCounter[pairIndx-1];
        if(sVPerThread<1)sVPerThread=1;
	sBlockShift=0;
	if(pairIndx>0)sBlockShift=PFSideCrossCounter[pairIndx-1];
    }
    
    __syncthreads();

  if(sEINum<=0){
	return;
	//Point In Polygon Test
  }
  else{
	  sStatus[tXIndx] = false;
	  int jStart;   
	  jStart=tXIndx==0?sVNum-1:tXIndx*sVPerThread-1;
	  mbr_t x1, x2, y1, y2;
	  for (int i=tXIndx*sVPerThread,  j=jStart; i<sVNum && i<(tXIndx+1)*sVPerThread; j=i++) {
            x1=sCoords[sShift+2*j];
            y1=sCoords[sShift+2*j+1];
            x2=sCoords[sShift+2*i];
            y2=sCoords[sShift+2*i+1];
	    if ( ((y2>sY0) != (y1>sY0)) && (sX0 < (x1-x2) * (sY0-y2) / (y1-y2) + x2) ){
	       sStatus[tXIndx] = !sStatus[tXIndx];
	       //printf("\nBL: %d\tPN: %d\tTX: %d\tStatus: %d\t i: %d\t sVPerThread:%d\t bIndx:%d\t oIndx:%d\n",blockIndx, pairIndx, tXIndx, sStatus[tXIndx],i,sVPerThread, bIndx, oIndx);
	    }
	  }
  }

  __syncthreads();
  
  int pw=2;
  while(blockDim.x/pw>0){
      if(tXIndx%pw==0 && tXIndx+pw/2<blockDim.x)
           if(sStatus[tXIndx+pw/2])sStatus[tXIndx]=!sStatus[tXIndx];        
      pw*=2;
      __syncthreads();
  }
    
  __syncthreads();
   
  if(tXIndx>=sEINum)return;

  //if(tXIndx==0 && sStatus[0]==1)printf("\nBL%d \t PN: %d \tChe Ajab!!!\n",blockIndx, pairIndx); 

  sCrossSideVector[sEIShift+tXIndx]+=(tXIndx%2==0)?!sStatus[0]:sStatus[0];  

  return;
}
//==============================================================================


//=========================== GenerateOutputPolygons ===========================
bool GetNextUnprocessedIntersection(bool* iProcessed, int* indx, int* indx2, int iNum){
   bool retVal=false;
   *indx2=-1;
   for(int i=*indx+1;i<iNum;i++){
     if(!iProcessed[i]){
        retVal=true;
        *indx=i;
        iProcessed[i]=true;
        if(i<iNum-1)*indx2=i+1;
        break;
     }
   }
return retVal;
}
//==============================================================================


//=========================== GenerateOutputPolygons ===========================
void GenerateOutputPolygons(int pairIndx, mbr_t* bCoords[], mbr_t* oCoords[], long bPolNum, long oPolNum, int* bVNum, int* oVNum, int* jCompactVector, double* bCrossSideVector, double* oCrossSideVector, int* b2oPointer, int* o2bPointer, int* SideCrossCounter, int* PFSideCrossCounter){
    int pairShift=0, iCurrentIndx, iNextIndx, oIndx, bIndx, oV, bV, writerIndx=0;      
    char outputPolys[300000];
    int eINum=SideCrossCounter[pairIndx];

    bool iProcessed[eINum];
    bIndx=jCompactVector[2*(pairIndx)];        
    oIndx=jCompactVector[2*(pairIndx)+1];        
    oV=oVNum[oIndx];
    bV=bVNum[bIndx];

    if(pairIndx>0)pairShift=PFSideCrossCounter[pairIndx-1];

    for(int i=0;i<eINum;i++)iProcessed[i]=false;

    int currentIndx=-1, nextIndx, iSwitch=0, *csPointer[2], csV[2], firstV, currentV;
    double *csVector[2];
    csPointer[0]=b2oPointer;
    csPointer[1]=o2bPointer;
    csVector[0]=bCrossSideVector;
    csVector[1]=oCrossSideVector;
    csV[0]=bV;
    csV[1]=oV;

    while(1){
       if(!GetNextUnprocessedIntersection(iProcessed, &currentIndx, &nextIndx, eINum))break;
       firstV=currentIndx;
       do{         
           writerIndx+=sprintf(&outputPolys[writerIndx], "\t:(%d)\t\n", currentIndx);

           //===================== Traversing all the way to the next intersection ========================
           int upperBound=(nextIndx!=-1)?nextIndx:csV[iSwitch];
           for(int i=currentIndx;i<upperBound; i++)writerIndx+=sprintf(&outputPolys[writerIndx], "\t:(%d)\t\n", i);
           if(!GetNextUnprocessedIntersection(iProcessed, &currentIndx, &nextIndx, eINum))break;	
           writerIndx+=sprintf(&outputPolys[writerIndx], "\t:(%d)<-->", currentIndx);
           //==============================================================================================
           
           //============================== Going to neighbor vertex ======================================
           currentIndx=*(csPointer[iSwitch]+pairShift+currentIndx);
           writerIndx+=sprintf(&outputPolys[writerIndx], "\t:(%d)\t\n", currentIndx);
           iSwitch=(iSwitch==0)?1:0;
           currentIndx=*(csPointer[iSwitch]+pairShift+currentIndx);
           //==============================================================================================
           if(iSwitch==0 && currentIndx==firstV)break;
       }while(1);
    }
    //printf("The clipped polygons for pair %d is :\n %s\n",pairIndx, outputPolys);
    return;
}
//==============================================================================


//=========================== EdgeIntersectCounter =============================
__global__ void EdgeIntersectCounter(mbr_t *bCoords, mbr_t * oCoords, long pairNum, int* jPSCounter, int* jCompactVector, int* bVNum, int* oVNum, int *bVPSNum, int* oVPSNum, int* actualEdgeCrossCounter){
    __shared__ int oIndx, bIndx, crossSidePerThread, oV, bV, bBase, oBase, crossCounter_PFS[1024];
    
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y, crossCounter;
    int blockIndx=bYIndx*gridDim.x+bXIndx, gtIndx=blockIndx*blockDim.x+tXIndx;        
    
    crossCounter=0;
    

    if(blockIndx>=pairNum)return;
    if(tXIndx==0){
        bBase=0;
        oBase=0;
        bIndx=jCompactVector[2*(blockIndx)];        
        oIndx=jCompactVector[2*(blockIndx)+1];        
        oV=oVNum[oIndx];
        bV=bVNum[bIndx];
        crossSidePerThread=bV*oV/blockDim.x;
        if(crossSidePerThread<1)crossSidePerThread=1;
        if(bIndx!=0)bBase=bVPSNum[bIndx-1];
        if(oIndx!=0)oBase=oVPSNum[oIndx-1];    
    }

    __syncthreads();   
    
    int bCoordIndx, oCoordIndx;
    if(tXIndx*crossSidePerThread<oV*bV){
	    bCoordIndx=tXIndx*crossSidePerThread/oV;
	    oCoordIndx=tXIndx*crossSidePerThread%oV;
	    int counter=0;
	    mbr_t bX1, bX2, bY1, bY2, oX1, oX2, oY1, oY2;
	    float m1, m2, x0, y0;
	    for(int i=bCoordIndx;i<bV && counter<crossSidePerThread;i++){
		int j;
		j=0;
		if(i==bCoordIndx)j=oCoordIndx;        
		bX1=bCoords[2*(bBase+i)];
		bY1=bCoords[2*(bBase+i)+1];
		if(i<bV-1){
			bX2=bCoords[2*(bBase+i+1)];
			bY2=bCoords[2*(bBase+i+1)+1];
		}
		else{
			bX2=bCoords[2*(bBase+0)];
			bY2=bCoords[2*(bBase+0)+1];
		}
		for(;j<oV && counter<crossSidePerThread;j++){
		    counter++;
		    oX1=oCoords[2*(oBase+j)];
		    oY1=oCoords[2*(oBase+j)+1];
		    if(j<oV-1){
			oX2=oCoords[2*(oBase+j+1)];
			oY2=oCoords[2*(oBase+j+1)+1];
		    }
		    else{
			oX2=oCoords[2*(oBase+0)];
			oY2=oCoords[2*(oBase+0)+1];
		    }
		    m1=(bY2-bY1)/(bX2-bX1);
		    m2=(oY2-oY1)/(oX2-oX1);
		    x0=((oY1-bY1)+m1*bX1-m2*oX1)/(m1-m2);
		    y0=m1*(x0-bX1)+bY1;
		    if( (bX1<=x0) && (x0<=bX2) && (oX1<=x0) && (x0<=oX2) && (bY1<=y0) && (y0<=bY2) && (oY1<=y0) && (y0<=oY2) ){
			crossCounter++;
		    }            
		}
	    }
    }
    crossCounter_PFS[tXIndx]=crossCounter;

    __syncthreads();
    
    int tempSum=0, p=1;
    while(p<blockDim.x){        
        if(tXIndx-p>=0){
            tempSum=crossCounter_PFS[tXIndx]+crossCounter_PFS[tXIndx-p];
        }                    
        __syncthreads();
        
        if(tXIndx-p>=0){
            crossCounter_PFS[tXIndx]=tempSum;
        }
        __syncthreads();
        
        p*=2;
    }

    if(tXIndx==0)actualEdgeCrossCounter[blockIndx]=crossCounter_PFS[1023];
    return;
    }
//==============================================================================


/*//============================== EdgeIntersectCounter =================================
__global__ void EdgeIntersectCounterLB(mbr_t *bCoords, mbr_t * oCoords, long pairNum, int* jPSCounter, int* jCompactVector, int* bVNum, int* oVNum, int *bPSVNum, int* oPSVNum, int* allEdgeCrossPSCounter, int * actualEdgeCrossCounter){    
    int baseEdgeCrossIndx=-1;
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y, oIndx, bIndx;
    int blockIndx=bYIndx*gridDim.x+bXIndx, gtIndx=blockIndx*blockDim.x+tXIndx;

    gtIndx*=EDGE_PER_THREAD;
    
    long allEdgeCrossCount=allEdgeCrossPSCounter[pairNum-1];
    
    if(gtIndx>=allEdgeCrossCount)return;
    
    long l1_Indx=0, l2_Indx=pairNum-1, mIndx=(l1_Indx+l2_Indx)/2;
    while(1){             
        if(l1_Indx>=allEdgeCrossCount)return;
        if(gtIndx+1<PFSideCrossCounter[mIndx] && mIndx==0){baseEdgeCrossIndx=0;break;}
        if(PFSideCrossCounter[mIndx-1]<gtIndx+1 && gtIndx+1<=PFSideCrossCounter[mIndx]){baseEdgeCrossIndx=mIndx;break;}
        if(PFSideCrossCounter[mIndx-1]>=gtIndx+1)l2_Indx=mIndx;
        if(PFSideCrossCounter[mIndx]<gtIndx+1)l1_Indx=mIndx;
        mIndx=(l1_Indx+l2_Indx)/2;        
    }
    oIndx=jCompactVector[2*matchedSideCrossIndx+1];
    bIndx=jCompactVector[2*matchedSideCrossIndx];

    
    int sideCrossNumForThisPol, oVN;
    oVN=oVNum[oIndx];

    sideCrossNumForThisPol=gtIndx;
    if(matchedSideCrossIndx!=0)sideCrossNumForThisPol-=PFSideCrossCounter[matchedSideCrossIndx-1];
    
    int bCoordIndx, oCoordIndx, bBase=0, oBase=0;
    if(bIndx!=0)bBase=bPFVNum[bIndx-1];
    if(oIndx!=0)oBase=oPFVNum[oIndx-1];    
    bCoordIndx=(sideCrossNumForThisPol)/oVN;
    oCoordIndx=(sideCrossNumForThisPol)%oVN;

    mbr_t bX1, bX2, bY1, bY2, oX1, oX2, oY1, oY2;
    bX1=bCoords[2*(bBase+bCoordIndx)];
    bY1=bCoords[2*(bBase+bCoordIndx)+1];
    bX2=bCoords[2*(bBase+bCoordIndx+1)];
    bY2=bCoords[2*(bBase+bCoordIndx+1)+1];
    oX1=oCoords[2*(oBase+oCoordIndx)];
    oY1=oCoords[2*(oBase+oCoordIndx)+1];
    oX2=oCoords[2*(oBase+oCoordIndx+1)];
    oY2=oCoords[2*(oBase+oCoordIndx+1)+1];
    
    float m1, m2, x0, y0;
    m1=(bY2-bY1)/(bX2-bX1);
    m2=(oY2-oY1)/(oX2-oX1);
    x0=((oY1-bY1)+m1*bX1-m2*oX1)/(m1-m2);
    y0=m1*(x0-bX1)+bY1;
    if( (bX1<=x0) && (x0<=bX2) && (oX1<=x0) && (x0<=oX2) && (bY1<=y0) && (y0<=bY2) && (oY1<=y0) && (y0<=oY2) ){        
        //sideIntersectionVector[2*gtIndx]=x0;
        //sideIntersectionVector[2*gtIndx+1]=y0;
        //if(x0==-1&&y0==-1)printf("(-1 , -1) is happened!");return;
    }
    else{        
        //sideIntersectionVector[2*gtIndx]=-1;
        //sideIntersectionVector[2*gtIndx+1]=-1;
    }
    return;
    }
//==============================================================================
*/
