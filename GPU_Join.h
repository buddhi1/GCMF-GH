
//============================== IntersectSide =================================
__global__ void UpdateBucketJoinFlag(mbr_t *bCoords, mbr_t * oCoords, long pairNum, long edgePairNum, int* jCompactVector, int* oVNum, long *bVPSNum, long* oVPSNum, long* allEdgeCrossPSCounter, char* bucketJoinFlag){    
    __shared__ long firstPairIndx;
    __shared__ char sJoinFlag[1024];
    __shared__ char pairShift[1024];

    int matchedPolyPairIndx=-1, tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y, oIndx, bIndx, blockIndx;
    long gtIndx;
    blockIndx=bYIndx*gridDim.x+bXIndx;
    gtIndx=blockIndx*blockDim.x+tXIndx;
    if(gtIndx>=edgePairNum)return;
    
//------------------------- Finding corresponding pair of edges ------------------------------
if(tXIndx==0 || tXIndx==1023){
    long l1_Indx=0, l2_Indx=pairNum-1, mIndx;
    mIndx=(l1_Indx+l2_Indx)/2;
    while(1){             
        if(gtIndx+1<=allEdgeCrossPSCounter[mIndx] && mIndx==0){matchedPolyPairIndx=0;break;}
        if(allEdgeCrossPSCounter[mIndx-1]<gtIndx+1 && gtIndx+1<=allEdgeCrossPSCounter[mIndx]){matchedPolyPairIndx=mIndx;break;}
        if(allEdgeCrossPSCounter[mIndx-1]>=gtIndx+1)l2_Indx=mIndx-1;
        if(allEdgeCrossPSCounter[mIndx]<gtIndx+1)l1_Indx=mIndx+1;
        mIndx=(l1_Indx+l2_Indx)/2; 
    }
    oIndx=jCompactVector[2*matchedPolyPairIndx+1];
    bIndx=jCompactVector[2*matchedPolyPairIndx];
    if(tXIndx==0)firstPairIndx=matchedPolyPairIndx;
}

__syncthreads();

matchedPolyPairIndx=firstPairIndx;
    oIndx=jCompactVector[2*matchedPolyPairIndx+1];
    bIndx=jCompactVector[2*matchedPolyPairIndx];
//--------------------------------------------------------------------------------------------

//----------------------------------- Intersection Test --------------------------------------
    int edgeCrossNumForThisPol, oVN;
    oVN=oVNum[oIndx];
    edgeCrossNumForThisPol=gtIndx;
    if(matchedPolyPairIndx!=0)edgeCrossNumForThisPol-=allEdgeCrossPSCounter[matchedPolyPairIndx-1];
    
    int bCoordIndx, oCoordIndx;
    long bBase=0, oBase=0;
    if(bIndx!=0)bBase=bVPSNum[bIndx-1];
    if(oIndx!=0)oBase=oVPSNum[oIndx-1];    
    bCoordIndx=(edgeCrossNumForThisPol)/oVN;
    oCoordIndx=(edgeCrossNumForThisPol)%oVN;


    mbr_t bX1, bX2, bY1, bY2, oX1, oX2, oY1, oY2;

    bX1=bCoords[2*(bBase+bCoordIndx)];
    bY1=bCoords[2*(bBase+bCoordIndx)+1];
    bX2=bCoords[2*(bBase+bCoordIndx+1)];
    bY2=bCoords[2*(bBase+bCoordIndx+1)+1];
    oX1=oCoords[2*(oBase+oCoordIndx)];
    oY1=oCoords[2*(oBase+oCoordIndx)+1];
    oX2=oCoords[2*(oBase+oCoordIndx+1)];
    oY2=oCoords[2*(oBase+oCoordIndx+1)+1];
   
    char tt=0; 
    coord_t m1, m2, x0, y0;
    if(bX2!=bX1)m1=(bY2-bY1)/(bX2-bX1);
    else m1=100000;
    if(oX2!=oX1)m2=(oY2-oY1)/(oX2-oX1);
    else m2=100000;

    if(m1!=m2){
       x0=((oY1-bY1)+m1*bX1-m2*oX1)/(m1-m2);
       y0=m1*(x0-bX1)+bY1;
       if( (bX1<=x0) && (x0<=bX2) && (oX1<=x0) && (x0<=oX2) && (bY1<=y0) && (y0<=bY2) && (oY1<=y0) && (y0<=oY2) ){        
           tt=1;
       }
    }
    
//--------------------------------------------------------------------------------------------
    pairShift[tXIndx]=matchedPolyPairIndx-firstPairIndx;
    sJoinFlag[tXIndx]=tt;

    int p=1, temp;
    while(p<blockDim.x){
      __syncthreads();
      if(tXIndx-p>=0 && pairShift[tXIndx]==pairShift[tXIndx-p]){
         temp=sJoinFlag[tXIndx]+sJoinFlag[tXIndx-p];
      }
      __syncthreads();
      if(tXIndx-p>=0 && pairShift[tXIndx]==pairShift[tXIndx-p]){
         sJoinFlag[tXIndx]=sJoinFlag[tXIndx]+sJoinFlag[tXIndx-p];
      } 
      p*=2;
    }

    if(tXIndx==blockDim.x-1 || pairShift[tXIndx]!=pairShift[tXIndx+1]);//bucketJoinFlag[gtIndx/ECC_THREAD_PER_BLOCK]=sJoinFlag[tXIndx];
    return;
    }
//==============================================================================


//=========================== EdgeIntersectCounter =============================
__global__ void EdgeIntersectCounter(mbr_t *bCoords, mbr_t * oCoords, long pairNum, int* jCompactVector, int* bVNum, int* oVNum, long *bPFVNum, long* oPFVNum, int* actualEdgeCrossCounter, coord_t* cMBR){
   __shared__ int oIndx, bIndx, crossSidePerThread, oV, bV, bBase, oBase, crossCounter_PFS[1024];
   __shared__ coord_t cX1, cX2, cY1, cY2; 
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
        cX1=cMBR[blockIndx*4];
        cY1=cMBR[blockIndx*4+1];
        cX2=cMBR[blockIndx*4+2];
        cY2=cMBR[blockIndx*4+3];
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
	    coord_t m1, m2, x0, y0;
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
        //------------------- Common MBR Filtering -----------------
                if(bX1<bX2){
                  if(bX1>cX2){counter+=oV-j;continue;}
                  if(bX2<cX1){counter+=oV-j;continue;}
                }
                else{
                  if(bX2>cX2){counter+=oV-j;continue;}
                  if(bX1<cX1){counter+=oV-j;continue;}
                }
                if(bY1<bY2){
                  if(bY1>cY2){counter+=oV-j;continue;}
                  if(bY2<cY1){counter+=oV-j;continue;}
                }
                else{
                  if(bY2>cY2){counter+=oV-j;continue;}
                  if(bY1<cY1){counter+=oV-j;continue;}
                }
        //----------------------------------------------------------
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
//=======================================================================================

//=========================== EdgeIntersectCounter2 =============================
__global__ void Count_EdgeIntersect(coord_t *bCoords, coord_t * oCoords, long pairNum, int* jCompactVector, int* jxy2IndexList, long *bPFVNum, long* oPFVNum, long* bEdgePSCounter, long* oEdgePSCounter, poly_size_t *bEdgeList, poly_size_t* oEdgeList, int* edgeIntersectCounter, int* jxyFlag){
   __shared__ int elementPerThread, bV, oV, bE, oE, crossCounter_PFS[1024];
   __shared__ long bBase, oBase, bEBase, oEBase;
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y, blockIndx, gtIndx;

    blockIndx=bYIndx*gridDim.x+bXIndx;
    gtIndx=blockIndx*blockDim.x+tXIndx;
    
    if(blockIndx>=pairNum)return;

    if(tXIndx==0){
        bBase=0;
        oBase=0;
        bEBase=0;
        oEBase=0;
        int indx=jxy2IndexList[blockIndx];
        int bIndx=jCompactVector[2*indx];        
        int oIndx=jCompactVector[2*indx+1];        
        if(bIndx!=0)bBase=bPFVNum[bIndx-1];
        if(oIndx!=0)oBase=oPFVNum[oIndx-1];    
        if(blockIndx!=0){
	   bEBase=bEdgePSCounter[blockIndx-1];    
	   oEBase=oEdgePSCounter[blockIndx-1];    
	}
        oV=oPFVNum[blockIndx]-oBase;
        bV=bPFVNum[blockIndx]-bBase;
        oE=oEdgePSCounter[blockIndx]-oEBase;
        bE=bEdgePSCounter[blockIndx]-bEBase;
        elementPerThread=bE*oE/blockDim.x;
        if(elementPerThread*blockDim.x<bE*oE)elementPerThread++;
    }

    __syncthreads(); 

    int bCoordIndx, oCoordIndx, intersectCounter=0;
    if(tXIndx*elementPerThread<oE*bE){
	    bCoordIndx=tXIndx*elementPerThread/oE;
	    oCoordIndx=tXIndx*elementPerThread%oE;
            if(bCoordIndx<bE)bCoordIndx=bEdgeList[bEBase+bCoordIndx]; 
            oCoordIndx=oEdgeList[oEBase+oCoordIndx]; 
	    int counter=0;
	    coord_t bX1, bX2, bY1, bY2, oX1, oX2, oY1, oY2;
	    for(int i=bCoordIndx;i<bE && counter<elementPerThread;i++){
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
		for(;j<oE && counter<elementPerThread;j++){
                    coord_t x0, y0, denom;
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
		    denom=(bX1-bX2)*(oY1-oY2)-(bY1-bY2)*(oX1-oX2);
                    if(denom==0)continue;
		    x0=((bX1*bY2-bY1*bX2)*(oX1-oX2)-(bX1-bX2)*(oX1*oY2-oY1*oX2))/denom;
                    if((x0>bX1 && x0>bX2)|| (x0<bX1 && x0<bX2) )continue;
                    if((x0>oX1 && x0>oX2)|| (x0<oX1 && x0<oX2) )continue;
		    y0=((bX1*bY2-bY1*bX2)*(oY1-oY2)-(bY1-bY2)*(oX1*oY2-oY1*oX2))/denom;
                    if((y0>bY1 && y0>bY2)|| (y0<bY1 && y0<bY2) )continue;
                    if((y0>oY1 && y0>oY2)|| (y0<oY1 && y0<oY2) )continue;
		    intersectCounter++;
		}
	    }
    }

    crossCounter_PFS[tXIndx]=intersectCounter;

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

    if(tXIndx==0){
	edgeIntersectCounter[blockIndx]=crossCounter_PFS[1023];
        jxyFlag[blockIndx]=crossCounter_PFS[1023]==0?0:1;
    }
    return;
    }
//=======================================================================================


//=========================== Segment_Intersect_Join =============================
__global__ void Segment_Intersect_Join(coord_t *bCoords, coord_t * oCoords, long pairNum, int* jCompactVector, int* jxy2IndexList, long *bPFVNum, long* oPFVNum, long* bEdgePSCounter, long* oEdgePSCounter, poly_size_t *bEdgeList, poly_size_t* oEdgeList, int* segIntersectJoinFlag){
   __shared__ int elementPerThread, bV, oV, bE, oE, joinFlag, bIndx, oIndx, mappedIndx;
   __shared__ long bBase, oBase, bEBase, oEBase;
   int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y, blockIndx, gtIndx;
    blockIndx=bYIndx*gridDim.x+bXIndx;
    gtIndx=blockIndx*blockDim.x+tXIndx;
    
    if(blockIndx>=pairNum)return;
    if(tXIndx==0){
        joinFlag=0;
        bBase=0;
        oBase=0;
        bEBase=0;
        oEBase=0;
        mappedIndx=jxy2IndexList[blockIndx];
        bIndx=jCompactVector[2*mappedIndx];        
        oIndx=jCompactVector[2*mappedIndx+1];        
        if(bIndx!=0)bBase=bPFVNum[bIndx-1];
        if(oIndx!=0)oBase=oPFVNum[oIndx-1];    
        if(mappedIndx!=0){
	   bEBase=bEdgePSCounter[mappedIndx-1];    
	   oEBase=oEdgePSCounter[mappedIndx-1];    
	}
        oV=oPFVNum[oIndx]-oBase;
        bV=bPFVNum[bIndx]-bBase;
        oE=oEdgePSCounter[mappedIndx]-oEBase;
        bE=bEdgePSCounter[mappedIndx]-bEBase;
        elementPerThread=bE*oE/blockDim.x;
        if(elementPerThread*blockDim.x<bE*oE)elementPerThread++;
    }

//if(tXIndx==0)printf("\n%d",elementPerThread);
//return;
    __syncthreads(); 

//if(blockIndx<15 && tXIndx==1)printf("\n%d:\t%d\t(%d,%d)\t(%d,%d)\t(%d,%d)\t %d %d %d %d\n", blockIndx, mappedIndx, bIndx, oIndx, bV, oV, bE, oE, bBase, oBase, bEBase, oEBase);

    int bCoordIndx, oCoordIndx, intersectCounter=0;
    if(tXIndx*elementPerThread<oE*bE){
	    bCoordIndx=tXIndx*elementPerThread/oE;
	    oCoordIndx=tXIndx*elementPerThread%oE;
	    int counter=0, j, bEIndx, oEIndx;
	    double bX1, bX2, bY1, bY2, oX1, oX2, oY1, oY2;
	    for(int i=bCoordIndx;i<bE && counter<elementPerThread && joinFlag==0;i++){
		j=0;
		if(i==bCoordIndx)j=oCoordIndx; 

                bEIndx=bEdgeList[bEBase+i];
                //bEIndx=i;

		bX1=bCoords[2*(bBase+bEIndx)];
		bY1=bCoords[2*(bBase+bEIndx)+1];
		if(bEIndx<bV-1){
			bX2=bCoords[2*(bBase+bEIndx+1)];
			bY2=bCoords[2*(bBase+bEIndx+1)+1];
		}
		else{
			bX2=bCoords[2*(bBase+0)];
			bY2=bCoords[2*(bBase+0)+1];
		}

    		if(bX1==bX2 && bY1==bY2){
                   continue;
                }
		for(;j<oE && counter<elementPerThread && joinFlag==0;j++){
		    counter++;

                    oEIndx=oEdgeList[oEBase+j];
                    //oEIndx=j;

		    oX1=oCoords[2*(oBase+oEIndx)];
		    oY1=oCoords[2*(oBase+oEIndx)+1];
		    if(oEIndx<oV-1){
			oX2=oCoords[2*(oBase+oEIndx+1)];
			oY2=oCoords[2*(oBase+oEIndx+1)+1];
		    }
		    else{
			oX2=oCoords[2*(oBase+0)];
			oY2=oCoords[2*(oBase+0)+1];
		    }

                    if(((bX2<oX1) && (bX2<oX2)) &&((bX1<oX1) && (bX1<oX2)) )continue;
                    if(((bX2>oX1) && (bX2>oX2)) &&((bX1>oX1) && (bX1>oX2)) )continue;
                    if(((bY2<oY1) && (bY2<oY2)) &&((bY1<oY1) && (bY1<oY2)) )continue;
                    if(((bY2>oY1) && (bY2>oY2)) &&((bY1>oY1) && (bY1>oY2)) )continue;

                    double x0, y0, denom;
		    denom=( (bX1-bX2) * (oY1-oY2)) - ((bY1-bY2) * (oX1-oX2));
                    if(denom==0){
                        if((oX1==oX2 && oX1==bX1) || (bX1==bX2 && bX1==oX1)){
			   joinFlag=1;
			}
                        else{
                           double oM=(oY2-oY1)/(oX2-oX1);
                           double bM=(bY2-bY1)/(bX2-bX1);
			   double oB=oY1-oM*oX1;
			   double bB=bY1-bM*bX1;
                           if(oB==bB)joinFlag=1;
                        }
                        continue;
                    }
                    double nomX = (((bX1*bY2)-(bY1*bX2)) * (oX1-oX2)) - ((bX1-bX2) * ((oX1*oY2)-(oY1*oX2)));
		    double nomY= ((((bX1*bY2)-(bY1*bX2)) * (oY1-oY2)) - (((bY1-bY2) * ((oX1*oY2)-(oY1*oX2)))));
		    //double nomX0= (((bX1*bY2)-(bY1*bX2)) * (oX1-oX2));
		    //double nomX1= ((bX1-bX2) * ((oX1*oY2)-(oY1*oX2)));
                    //nomX=nomX0-nomX1;
                    x0=nomX/denom;
                    y0=nomY/denom;
     
//if(bIndx==1629 && oIndx==7 && bEIndx==719)printf("\n[%d,%d]\t(%.12f,%.12f)(%.12f %.12f)\t%.12lf %.12lf\t(%.12f,%.12f)(%.12f,%.12f)", bEIndx, oEIndx, bX1,bY1,bX2,bY2,x0,y0,oX1,oY1,oX2,oY2);
//if(bIndx==1629 && oIndx==7)printf("\n%d: [%d,%d]", blockIndx, bEIndx, oEIndx);

                    if((x0>bX1 && x0>bX2)|| (x0<bX1 && x0<bX2) )continue;
                    if((x0>oX1 && x0>oX2)|| (x0<oX1 && x0<oX2) )continue;
                    if((y0>bY1 && y0>bY2)|| (y0<bY1 && y0<bY2) )continue;
                    if((y0>oY1 && y0>oY2)|| (y0<oY1 && y0<oY2) )continue;

//if(bIndx==1629 && oIndx==7)printf("\n[%d,%d]\t(%.12f,%.12f)(%.12f %.12f)\t%.12lf %.12lf\t(%.12f,%.12f)(%.12f,%.12f)", bEIndx, oEIndx, bX1,bY1,bX2,bY2,x0,y0,oX1,oY1,oX2,oY2);

                    joinFlag=1;		   
		}
	    }
    }

    __syncthreads();

    if(tXIndx==0){
        segIntersectJoinFlag[blockIndx]=joinFlag;
    }
    return;
  }
//=======================================================================================


//=========================== Get_CMBR =============================
__global__ void Get_CMBR(long pairNum, int* jCompactVector, int* piPFlag, char* piPType, coord_t* bMBR, coord_t* oMBR, coord_t* cMBR, int *joinFlag){
    int bIndx, oIndx, tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y;
    int blockIndx=bYIndx*gridDim.x+bXIndx;        
    blockIndx=blockIndx*blockDim.x+tXIndx;

    if(blockIndx>=pairNum)return;

    bIndx=jCompactVector[2*(blockIndx)];        
    oIndx=jCompactVector[2*(blockIndx)+1]; 

    __syncthreads();

    coord_t cX1, cX2, cY1, cY2; 
    coord_t x1, x2, a1, a2, y1, y2, b1, b2;
    x1=bMBR[bIndx*4];
    x2=bMBR[bIndx*4+2];
    y1=bMBR[bIndx*4+1];
    y2=bMBR[bIndx*4+3];
    a1=oMBR[oIndx*4];
    a2=oMBR[oIndx*4+2];
    b1=oMBR[oIndx*4+1];
    b2=oMBR[oIndx*4+3];
    cX1=(a1>x1?a1:x1);
    cMBR[4*blockIndx]=cX1;
    cX2=(a2<x2?a2:x2);
    cMBR[4*blockIndx+2]=cX2;
    cY1=(b1>y1?b1:y1);
    cMBR[4*blockIndx+1]=cY1;
    cY2=(b2<y2?b2:y2);
    cMBR[4*blockIndx+3]=cY2;
    joinFlag[blockIndx]=1;
    //-----------------------------------------------------------------------------------------------------------
    //If common MBR is equal to one of the MBRs, the other MBR may contain this one. (Point in Polygon is needed)
    //piPType[i] = 1 : The overlay polygon may contain the base polygon  |   = 2 : The base polygon may contain the overlay polygon
    if(x1==cX1 && x2==cX2 && y1==cY1 && y2==cY2){
       piPFlag[blockIndx]=1;
       piPType[blockIndx]=1;
    }
    else if(a1==cX1 && a2==cX2 && b1==cY1 && b2==cY2){
      piPFlag[blockIndx]=1;
      piPType[blockIndx]=2;
    }        
    else{
      piPFlag[blockIndx]=0;
      piPType[blockIndx]=0;
    }
    return;
}

//=========================== Count_CMF =============================
__global__ void Count_CMF(coord_t *bCoords, coord_t * oCoords, coord_t* cMBR, long pairNum2, int* jCompactVector, int* jTempIndexList, int* bVNum, int* oVNum, long *bPFVNum, long* oPFVNum, int* bEdgeCounter, int* oEdgeCounter, int* joinFlag){
   __shared__ int mappedIndx, bIndx, oIndx, edgePerThread[2], vNum[2], baseCoord[2], edgeCounter_PFS[2][512], *edgeCounter[2];
   __shared__ coord_t cX1, cX2, cY1, cY2; 
   __shared__ coord_t *coords[2];
    int tXIndx=threadIdx.x, tYIndx=threadIdx.y, bXIndx=blockIdx.x, bYIndx=blockIdx.y, counter;
    int blockIndx=bYIndx*gridDim.x+bXIndx;        
    counter=0;

    if(blockIndx>=pairNum2)return;

    if(tXIndx==0){
       if(tYIndx==0){
          mappedIndx=jTempIndexList[blockIndx];
          oIndx=jCompactVector[2*mappedIndx+1]; 
          bIndx=jCompactVector[2*mappedIndx];        
       }
    }

    __syncthreads();


    if(tXIndx==0){
       if(tYIndx==0){
          cX1=cMBR[4*mappedIndx];
          cY1=cMBR[4*mappedIndx+1];
          cX2=cMBR[4*mappedIndx+2];
          cY2=cMBR[4*mappedIndx+3];
       } 
       else{
          vNum[0]=bVNum[bIndx];
          vNum[1]=oVNum[oIndx];
          coords[0]=bCoords;
          coords[1]=oCoords;
          baseCoord[0]=0;
          baseCoord[1]=0;
          if(oIndx!=0)baseCoord[1]=oPFVNum[oIndx-1];
          if(bIndx!=0)baseCoord[0]=bPFVNum[bIndx-1];
          edgeCounter[0]=bEdgeCounter;
          edgeCounter[1]=oEdgeCounter;
       }
   }

   __syncthreads();   

   if(tXIndx==0){
       edgePerThread[tYIndx]=vNum[tYIndx]/blockDim.x;
       if(edgePerThread[tYIndx]*blockDim.x<vNum[tYIndx])edgePerThread[tYIndx]++;
   }

    __syncthreads();   
 
    if(tXIndx*edgePerThread[tYIndx]<vNum[tYIndx]){
	    coord_t x1, x2, y1, y2;
	    for(int i=tXIndx*edgePerThread[tYIndx];i<(tXIndx+1)*edgePerThread[tYIndx] && i<vNum[tYIndx];i++){
		x1=coords[tYIndx][2*(baseCoord[tYIndx]+i)];
		y1=coords[tYIndx][2*(baseCoord[tYIndx]+i)+1];
		if(i<vNum[tYIndx]-1){
		   x2=coords[tYIndx][2*(baseCoord[tYIndx]+i+1)];
		   y2=coords[tYIndx][2*(baseCoord[tYIndx]+i+1)+1];
		}
		else{
		   x2=coords[tYIndx][2*(baseCoord[tYIndx]+0)];
		   y2=coords[tYIndx][2*(baseCoord[tYIndx]+0)+1];
		}
        //------------------- Common MBR Filtering -----------------
                if( (cX1>x1 && cX1>x2) || (cX2<x1 && cX2<x2) )continue;
                if( (cY1>y1 && cY1>y2) || (cY2<y1 && cY2<y2) )continue;
                counter++;
        //----------------------------------------------------------
          }
    }
    edgeCounter_PFS[tYIndx][tXIndx]=counter;

    __syncthreads();
    
    int tempSum=0, p=1;
    while(p<blockDim.x){        
        if(tXIndx-p>=0){
           tempSum=edgeCounter_PFS[tYIndx][tXIndx]+edgeCounter_PFS[tYIndx][tXIndx-p];
        }                    
        __syncthreads();
        
        if(tXIndx-p>=0){
           edgeCounter_PFS[tYIndx][tXIndx]=tempSum;
        }
        __syncthreads();
        
        p*=2;
    }

    __syncthreads();


    if(tXIndx==0 && tYIndx==0){
	if(edgeCounter_PFS[0][blockDim.x-1]==0 || edgeCounter_PFS[1][blockDim.x-1]==0){
	   joinFlag[mappedIndx]=0;
	   edgeCounter[0][mappedIndx]=0;
	   edgeCounter[1][mappedIndx]=0;
	}
	else{
	   edgeCounter[0][mappedIndx]=edgeCounter_PFS[0][blockDim.x-1];
	   edgeCounter[1][mappedIndx]=edgeCounter_PFS[1][blockDim.x-1];
	}
        //edgeCounter[0][mappedIndx]=vNum[0];
	//edgeCounter[1][mappedIndx]=vNum[1];
    }
    return;
    }
//=======================================================================================


//=========================== Apply_CMF =============================
__global__ void Apply_CMF(coord_t *bCoords, coord_t * oCoords, long pairNum, int* jCompactVector, int *jxy2IndexList, int* bVNum, int* oVNum, long *bPFVNum, long* oPFVNum, coord_t* cMBR, long* bEdgePSCounter, long* oEdgePSCounter, poly_size_t *bEdgeList, poly_size_t *oEdgeList){
   __shared__ int bIndx, oIndx, edgePerThread[2], vNum[2], baseCoord[2], edgeCounter_PFS[2][512], mappedIndx;
   __shared__ poly_size_t *edgeList[2];
   __shared__ long edgeListBase[2];
   __shared__ coord_t cX1, cX2, cY1, cY2; 
   __shared__ coord_t *coords[2];
   __shared__ char retStatus;
    int tXIndx=threadIdx.x, tYIndx=threadIdx.y, bXIndx=blockIdx.x, bYIndx=blockIdx.y, counter=0;
    int blockIndx=bYIndx*gridDim.x+bXIndx;        

    if(blockIndx>=pairNum)return;
  
    if(tXIndx==0){
       if(tYIndx==0){
          //int indx=jxy2IndexList[blockIndx];
          mappedIndx=jxy2IndexList[blockIndx];
          oIndx=jCompactVector[2*mappedIndx+1]; 
          bIndx=jCompactVector[2*mappedIndx];        
       }
    }

    __syncthreads();

    if(tXIndx==0){
       if(tYIndx==0){
          cX1=cMBR[4*mappedIndx];
          cY1=cMBR[4*mappedIndx+1];
          cX2=cMBR[4*mappedIndx+2];
          cY2=cMBR[4*mappedIndx+3];
          vNum[0]=bVNum[bIndx];
          coords[0]=bCoords;
          baseCoord[0]=0;
          if(bIndx!=0)baseCoord[0]=bPFVNum[bIndx-1];
          edgeListBase[0]=0;
          if(mappedIndx>0)edgeListBase[0]=bEdgePSCounter[mappedIndx-1];
          edgeList[0]=bEdgeList;
       }
       else{
          vNum[1]=oVNum[oIndx];
          coords[1]=oCoords;
          baseCoord[1]=0;
          if(oIndx!=0)baseCoord[1]=oPFVNum[oIndx-1];
          edgeListBase[1]=0;
          if(mappedIndx>0)edgeListBase[1]=oEdgePSCounter[mappedIndx-1];
          edgeList[1]=oEdgeList;
       }

       edgePerThread[tYIndx]=vNum[tYIndx]/blockDim.x;
       if(edgePerThread[tYIndx]*blockDim.x<vNum[tYIndx])edgePerThread[tYIndx]++;
    }

    __syncthreads();   

    int edgeTempList[400];
    if(tXIndx*edgePerThread[tYIndx]<vNum[tYIndx]){
	    coord_t x1, x2, y1, y2;
	    for(int i=tXIndx*edgePerThread[tYIndx];i<(tXIndx+1)*edgePerThread[tYIndx] && i<vNum[tYIndx];i++){
		x1=coords[tYIndx][2*(baseCoord[tYIndx]+i)];
		y1=coords[tYIndx][2*(baseCoord[tYIndx]+i)+1];
		if(i<vNum[tYIndx]-1){
		   x2=coords[tYIndx][2*(baseCoord[tYIndx]+i+1)];
		   y2=coords[tYIndx][2*(baseCoord[tYIndx]+i+1)+1];
		}
		else{
		   x2=coords[tYIndx][2*(baseCoord[tYIndx]+0)];
		   y2=coords[tYIndx][2*(baseCoord[tYIndx]+0)+1];
		}
        //------------------- Common MBR Filtering -----------------

                if( (cX1>x1 && cX1>x2) || (cX2<x1 && cX2<x2) )continue;
                if( (cY1>y1 && cY1>y2) || (cY2<y1 && cY2<y2) )continue;
 
                edgeTempList[counter++]=i;
                //if(counter>=10)printf("\nOverflow error in number of edges per thread!%d:%d:%d\n", blockIndx, tYIndx, tXIndx);
        //----------------------------------------------------------
          }
    }
    edgeCounter_PFS[tYIndx][tXIndx]=counter;

    __syncthreads();
    
    int tempSum=0, p=1;
    while(p<blockDim.x){        
        if(tXIndx-p>=0){
           tempSum=edgeCounter_PFS[tYIndx][tXIndx]+edgeCounter_PFS[tYIndx][tXIndx-p];
        }                    
        __syncthreads();
        
        if(tXIndx-p>=0){
           edgeCounter_PFS[tYIndx][tXIndx]=tempSum;
        }
        __syncthreads();
        
        p*=2;
    }

    __syncthreads();

    if(counter>=400){printf("\nedge list memory overflow at : %d:%d.%d with %d\n", mappedIndx, tYIndx, tXIndx, counter);return;}
    int shift=0;
    if(tXIndx>0)shift=edgeCounter_PFS[tYIndx][tXIndx-1];



//    if(tXIndx==0 && tYIndx==0 && edgeCounter_PFS[0][blockDim.x-1]!=bEdgePSCounter[mappedIndx]-edgeListBase[0]){
//printf("\nCount error at :%d : %d\t %d  <>   %d\n", blockIndx, mappedIndx, edgeCounter_PFS[0][blockDim.x-1], bEdgePSCounter[mappedIndx]-edgeListBase[0]);return;}




    for(int i=0;i<counter;i++)edgeList[tYIndx][edgeListBase[tYIndx]+shift+i]=edgeTempList[i];
    return;
    }
//=======================================================================================

//=========================== Count_CMF_1D =============================
__global__ void Count_CMF_1D(coord_t *bCoords, coord_t * oCoords, coord_t* cMBR, long pairNum2, int* jCompactVector, int* jTempIndexList, int* bVNum, int* oVNum, long *bPFVNum, long* oPFVNum, int* bEdgeCounter, int* oEdgeCounter, int* joinFlag){
   __shared__ int mappedIndx, pIndx, edgePerThread, vNum, baseCoord, edgeCounter_PFS[1024], *edgeCounter;
   __shared__ coord_t cX1, cX2, cY1, cY2; 
   __shared__ coord_t *coords;
    int tXIndx=threadIdx.x, tYIndx=threadIdx.y, bXIndx=blockIdx.x, bYIndx=blockIdx.y, counter;
    int blockIndx=bYIndx*gridDim.x+bXIndx;        
    counter=0;

    if(blockIndx>=2*pairNum2)return;

    if(tXIndx==0){
      mappedIndx=jTempIndexList[blockIndx/2];
      cX1=cMBR[4*mappedIndx];
      cY1=cMBR[4*mappedIndx+1];
      cX2=cMBR[4*mappedIndx+2];
      cY2=cMBR[4*mappedIndx+3];
 
       baseCoord=0;
       if(blockIndx%2==0){
          pIndx=jCompactVector[2*mappedIndx];        
          vNum=bVNum[pIndx];
          coords=bCoords;
          if(pIndx!=0)baseCoord=bPFVNum[pIndx-1];
          edgeCounter=bEdgeCounter;
       }
       else{
          pIndx=jCompactVector[2*mappedIndx+1]; 
          vNum=oVNum[pIndx];
          coords=oCoords;
          if(pIndx!=0)baseCoord=oPFVNum[pIndx-1];
          edgeCounter=oEdgeCounter;
       }
    }

   __syncthreads();   

   if(tXIndx==0){
       edgePerThread=vNum/blockDim.x;
       if(edgePerThread*blockDim.x<vNum)edgePerThread++;
   }

    __syncthreads();   
 
    if(tXIndx*edgePerThread<vNum){
	    coord_t x1, x2, y1, y2;
	    for(int i=tXIndx*edgePerThread;i<(tXIndx+1)*edgePerThread && i<vNum;i++){
		x1=coords[2*(baseCoord+i)];
		y1=coords[2*(baseCoord+i)+1];
		if(i<vNum-1){
		   x2=coords[2*(baseCoord+i+1)];
		   y2=coords[2*(baseCoord+i+1)+1];
		}
		else{
		   x2=coords[2*(baseCoord+0)];
		   y2=coords[2*(baseCoord+0)+1];
		}
        //------------------- Common MBR Filtering -----------------
                if( (cX1>x1 && cX1>x2) || (cX2<x1 && cX2<x2) )continue;
                if( (cY1>y1 && cY1>y2) || (cY2<y1 && cY2<y2) )continue;
                counter++;
        //----------------------------------------------------------
          }
    }
    edgeCounter_PFS[tXIndx]=counter;

    __syncthreads();
    
    int tempSum=0, p=1;
    while(p<blockDim.x){        
        if(tXIndx-p>=0){
           tempSum=edgeCounter_PFS[tXIndx]+edgeCounter_PFS[tXIndx-p];
        }                    
        __syncthreads();
        
        if(tXIndx-p>=0){
           edgeCounter_PFS[tXIndx]=tempSum;
        }
        __syncthreads();
        
        p*=2;
    }

    __syncthreads();


    if(tXIndx==0){
	edgeCounter[mappedIndx]=edgeCounter_PFS[blockDim.x-1];
	if(edgeCounter_PFS[blockDim.x-1]==0){
	   joinFlag[mappedIndx]=0;
	}
        //edgeCounter[0][mappedIndx]=vNum[0];
	//edgeCounter[1][mappedIndx]=vNum[1];
    }
    return;
    }
//=======================================================================================



//============================ PointInPolygonTest ==============================
__global__ void Point_In_Polygon_Test(coord_t* bCoords, coord_t* oCoords, long pipNum, int* jxyVector, int* jPiPIndexList, char* pipType, long* bVPSNum, long* oVPSNum, int* pipJoinFlag, int* joinFlag){
   __shared__ int bIndx, oIndx, pointPerThread, vNum, mappedIndx;
   __shared__ char cPS[1024];
   __shared__ coord_t pX, pY, *polyCoord; 
    int tXIndx=threadIdx.x, tYIndx=threadIdx.y, bXIndx=blockIdx.x, bYIndx=blockIdx.y, counter=0;
    int blockIndx=bYIndx*gridDim.x+bXIndx;        

    if(blockIndx>=pipNum)return;
    if(tXIndx==0){
       if(tYIndx==0){
          mappedIndx=jPiPIndexList[blockIndx];
          oIndx=jxyVector[2*mappedIndx+1]; 
          bIndx=jxyVector[2*mappedIndx];        
       }
    }
    __syncthreads();

    if(tXIndx==0){
       long bShift=0, oShift=0;
       if(bIndx>0)bShift=bVPSNum[bIndx-1];
       if(oIndx>0)oShift=oVPSNum[oIndx-1];

       if(pipType[mappedIndx]==1){
         pX=*(bCoords+2*bShift);
         pY=*(bCoords+2*bShift+1);
         vNum=oVPSNum[oIndx]-oShift;
	 polyCoord=oCoords+2*oShift;
       }
       else if(pipType[mappedIndx]==2){
         pX=*(oCoords+2*oShift);
         pY=*(oCoords+2*oShift+1);
         vNum=bVPSNum[bIndx]-bShift;
	 polyCoord=bCoords+2*bShift;
       }
       else printf("\nERROR! pipType value is invalid at %d of (%d,%d)\n", blockIndx, bIndx, oIndx);
  
       pointPerThread=vNum/blockDim.x;
       if(pointPerThread*blockDim.x<vNum)pointPerThread++;
    }

    __syncthreads();   
    
  
    int i, j, c=0;
    coord_t X1, Y1, X2, Y2;
    i=tXIndx*pointPerThread;
    j=tXIndx==0?vNum-1:i-1;



//int bTest=1;
//if(blockIndx==bTest && tXIndx==0)printf("\nbIndx=%d\toIndx=%d\t (%f,%f)\tvNum=%d\tPPT=%d\n", bIndx, oIndx, pX, pY, vNum, pointPerThread);
//    __syncthreads();




    for(;i<(tXIndx+1)*pointPerThread && i<vNum;j=i++){
       Y1=*(polyCoord+2*i+1);
       Y2=*(polyCoord+2*j+1);
       X1=*(polyCoord+2*i);
       X2=*(polyCoord+2*j);

//if(bIndx==1486 && oIndx==11)printf("\nBBBB: %d\n", blockIndx);

//if(blockIndx==bTest){
//printf("\n %d:(%f,%f) %d", i, X1, Y1, tXIndx);
//printf("\n %d:(%f,%f) %d", i, X1, Y1, tXIndx);
//}



       if((Y1>pY)==(Y2>pY))continue;
       if(pX<(X2-X1)*(pY-Y1)/(Y2-Y1) + X1){c=!c;}
      

    }
    cPS[tXIndx]=c;

    __syncthreads();

    int tempSum=0, p=1;
    while(p<blockDim.x){        
        if(tXIndx-p>=0){
           tempSum=cPS[tXIndx-p]==0?cPS[tXIndx]:!cPS[tXIndx];
        }
        __syncthreads();
        
        if(tXIndx-p>=0){
           cPS[tXIndx]=tempSum;
        }
        __syncthreads();
        
        p*=2;
    }

    __syncthreads();

    if(tXIndx==0){
	pipJoinFlag[mappedIndx]=cPS[blockDim.x-1];
        if(cPS[blockDim.x-1]==1)joinFlag[mappedIndx]=0;
    }

    return;
}
//==============================================================================


//=========================== Segment_Intersect_Join =============================
__global__ void Segment_Intersect_Join2(coord_t *bCoords, coord_t * oCoords, long pairNum, int* jCompactVector, int* jxy2IndexList, long *bPFVNum, long* oPFVNum, long* bEdgePSCounter, long* oEdgePSCounter, long *workLoadPSCounter, long workLoadNum, poly_size_t *bEdgeList, poly_size_t* oEdgeList, int* joinFlag){

   __shared__ char pairID[512];

   int bV, oV, bE, oE, bIndx, oIndx, mappedIndx;
   long bBase, oBase, bEBase, oEBase, gtIndx;
   int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y, blockIndx;
    blockIndx=bYIndx*gridDim.x+bXIndx;
    gtIndx=(blockIndx*blockDim.x+tXIndx)*EDGE_PER_THREAD;
    
    if(gtIndx>=workLoadNum)return;

    long lPairIndx=-1, uPairIndx=-1, l1_Indx=0, l2_Indx=pairNum-1, mIndx=(l1_Indx+l2_Indx)/2;
    while(1){             
        mappedIndx=jxy2IndexList[mIndx];
        if(gtIndx+1<workLoadPSCounter[mappedIndx] && mappedIndx==0){lPairIndx=0;break;}
        if(workLoadPSCounter[mappedIndx-1]<gtIndx+1 && gtIndx+1<=workLoadPSCounter[mappedIndx]){lPairIndx=mIndx;break;}
        if(mIndx==l1_Indx){lPairIndx=l2_Indx;break;}
        if(workLoadPSCounter[mappedIndx-1]>=gtIndx+1)l2_Indx=mIndx;
        if(workLoadPSCounter[mappedIndx]<gtIndx+1)l1_Indx=mIndx;
        mIndx=(l1_Indx+l2_Indx)/2;        
    }
    l1_Indx=0;
    l2_Indx=pairNum-1;
    mIndx=(l1_Indx+l2_Indx)/2;
    gtIndx+=EDGE_PER_THREAD;
    while(1){             
        mappedIndx=jxy2IndexList[mIndx];
        if(gtIndx+1<workLoadPSCounter[mappedIndx] && mappedIndx==0){uPairIndx=0;break;}
        if(workLoadPSCounter[mappedIndx-1]<gtIndx+1 && gtIndx+1<=workLoadPSCounter[mappedIndx]){uPairIndx=mIndx;break;}
        if(mIndx==l1_Indx){uPairIndx=l2_Indx;break;}
        if(workLoadPSCounter[mappedIndx-1]>=gtIndx+1)l2_Indx=mIndx;
        if(workLoadPSCounter[mappedIndx]<gtIndx+1)l1_Indx=mIndx;
        mIndx=(l1_Indx+l2_Indx)/2;        
    }

    gtIndx-=EDGE_PER_THREAD;
    int counter=0, ePairIndx;
    long loadShift=lPairIndx==0?0:workLoadPSCounter[lPairIndx-1];
    ePairIndx=gtIndx-loadShift;
    for(int k=lPairIndx;k<=uPairIndx && counter<EDGE_PER_THREAD;k++){
        mappedIndx=jxy2IndexList[k];
        bBase=0;
        oBase=0;
        bEBase=0;
        oEBase=0;
        bIndx=jCompactVector[2*mappedIndx];        
        oIndx=jCompactVector[2*mappedIndx+1];        
        if(bIndx!=0)bBase=bPFVNum[bIndx-1];
        if(oIndx!=0)oBase=oPFVNum[oIndx-1];    
        if(mappedIndx!=0){
	   bEBase=bEdgePSCounter[mappedIndx-1];    
	   oEBase=oEdgePSCounter[mappedIndx-1];    
	}
        oV=oPFVNum[oIndx]-oBase;
        bV=bPFVNum[bIndx]-bBase;
        oE=oEdgePSCounter[mappedIndx]-oEBase;
        bE=bEdgePSCounter[mappedIndx]-bEBase;
        int bCoordIndx, oCoordIndx;

	bCoordIndx=ePairIndx/oE;
	oCoordIndx=ePairIndx%oE;
	int j, bEIndx, oEIndx, tCounter;
	double bX1, bX2, bY1, bY2, oX1, oX2, oY1, oY2;
        tCounter=0;
	for(int i=bCoordIndx;i<bE && counter+tCounter<EDGE_PER_THREAD && joinFlag[k]==0;i++){
	    j=0;
	    if(i==bCoordIndx)j=oCoordIndx; 
            bEIndx=bEdgeList[bEBase+i];
	    bX1=bCoords[2*(bBase+bEIndx)];
	    bY1=bCoords[2*(bBase+bEIndx)+1];
	    if(bEIndx<bV-1){
		bX2=bCoords[2*(bBase+bEIndx+1)];
		bY2=bCoords[2*(bBase+bEIndx+1)+1];
	    }
	    else{
		bX2=bCoords[2*(bBase+0)];
		bY2=bCoords[2*(bBase+0)+1];
	    }
    	    if(bX1==bX2 && bY1==bY2){tCounter+=oE;continue;}
	    for(;j<oE && counter+tCounter<EDGE_PER_THREAD && joinFlag[k]==0;j++){
	        tCounter++;
                oEIndx=oEdgeList[oEBase+j];
		oX1=oCoords[2*(oBase+oEIndx)];
		oY1=oCoords[2*(oBase+oEIndx)+1];
		if(oEIndx<oV-1){
		   oX2=oCoords[2*(oBase+oEIndx+1)];
		   oY2=oCoords[2*(oBase+oEIndx+1)+1];
		}
		else{
		   oX2=oCoords[2*(oBase+0)];
		   oY2=oCoords[2*(oBase+0)+1];
		}
                if(((bX2<oX1) && (bX2<oX2)) &&((bX1<oX1) && (bX1<oX2)) )continue;
                if(((bX2>oX1) && (bX2>oX2)) &&((bX1>oX1) && (bX1>oX2)) )continue;
                if(((bY2<oY1) && (bY2<oY2)) &&((bY1<oY1) && (bY1<oY2)) )continue;
                if(((bY2>oY1) && (bY2>oY2)) &&((bY1>oY1) && (bY1>oY2)) )continue;

                double x0, y0, denom;
	        denom=( (bX1-bX2) * (oY1-oY2)) - ((bY1-bY2) * (oX1-oX2));
                if(denom==0){
                   if((oX1==oX2 && oX1==bX1) || (bX1==bX2 && bX1==oX1)){
		      joinFlag[k]=1;
	    	   }
                   else{
                       double oM=(oY2-oY1)/(oX2-oX1);
                       double bM=(bY2-bY1)/(bX2-bX1);
 	    	       double oB=oY1-oM*oX1;
	  	       double bB=bY1-bM*bX1;
                       if(oB==bB)joinFlag[k]=1;
                    }
                    continue;
                 }
                 double nomX = (((bX1*bY2)-(bY1*bX2)) * (oX1-oX2)) - ((bX1-bX2) * ((oX1*oY2)-(oY1*oX2)));
 	         double nomY= ((((bX1*bY2)-(bY1*bX2)) * (oY1-oY2)) - (((bY1-bY2) * ((oX1*oY2)-(oY1*oX2)))));
                 x0=nomX/denom;
                 y0=nomY/denom;
     
                 if((x0>bX1 && x0>bX2)|| (x0<bX1 && x0<bX2) )continue;
                 if((x0>oX1 && x0>oX2)|| (x0<oX1 && x0<oX2) )continue;
                 if((y0>bY1 && y0>bY2)|| (y0<bY1 && y0<bY2) )continue;
                 if((y0>oY1 && y0>oY2)|| (y0<oY1 && y0<oY2) )continue;

                 joinFlag[k]=1;		   
		}
	    }
            counter+=oE*(bE-bCoordIndx)-oCoordIndx;
            ePairIndx=0;
      }
    return;
  }
//=======================================================================================
