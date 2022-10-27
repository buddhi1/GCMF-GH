

//=============================== DotProduct ================================
__global__ void GPUDotProduct(long elementNum, int* vector1, char* vector2, int* result){
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y;
    
    long blockIndx=bYIndx*gridDim.x+bXIndx;
    long elementIndx=blockIndx*blockDim.x+tXIndx;
    if(elementIndx<elementNum){
         result[elementIndx]=vector1[elementIndx]*vector2[elementIndx];
    }
    return;
}
//==============================================================================


//=============================== DotProduct ================================
__global__ void GPUDotProduct(long elementNum, int* vector1, int* vector2, int* result){
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y;
    
    long blockIndx=bYIndx*gridDim.x+bXIndx;
    long elementIndx=blockIndx*blockDim.x+tXIndx;
    if(elementIndx<elementNum){
         result[elementIndx]=vector1[elementIndx]*vector2[elementIndx];
    }
    return;
}
//==============================================================================


//=============================== Initialize_Vector  ================================
__global__ void Initialize_Vector(long elementNum, int* vector1, int val){
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y;
    
    int blockIndx=bYIndx*gridDim.x+bXIndx;
    int elementIndx=blockIndx*blockDim.x+tXIndx;
    if(elementIndx<elementNum){
         vector1[elementIndx]=val;
    }
    return;
}
//==============================================================================


//=============================== Initialize_Vector  ================================
__global__ void Initialize_Vector2(long elementNum, int* vector1, int* vector2, int val){
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y;
    
    int blockIndx=bYIndx*gridDim.x+bXIndx;
    int elementIndx=blockIndx*blockDim.x+tXIndx;
    if(elementIndx<elementNum){
         vector1[elementIndx]=val;
         vector2[elementIndx]=val;
    }
    return;
}
//==============================================================================


//=============================== Compact_Bucket_Vector ================================
__global__ void Compact_Bucket_Vector(long elementNum, long* vPSCounter, int* elementVector, int *compactVector, char bucketWidth){
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y;
    int blockIndx=bYIndx*gridDim.x+bXIndx;
    int eIndx=blockIndx*blockDim.x+tXIndx, baseIndx=0;
    if(eIndx>=elementNum||eIndx==0)return;

    baseIndx=vPSCounter[eIndx-1];
    if(vPSCounter[eIndx]==baseIndx)return;
    for(int i=0;i<bucketWidth;i++)compactVector[bucketWidth*baseIndx+i]=elementVector[bucketWidth*eIndx+i];
    }
//==============================================================================


//=============================== Compact_CMF_Vector ================================
__global__ void Compact_CMF_Vector2(long elementNum, long* vPSCounter, int* elementVector, int *compactVector){
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y;
    
    int blockIndx=bYIndx*gridDim.x+bXIndx;
    int eIndx=blockIndx*blockDim.x+tXIndx;
    int baseIndx=0;
    if(eIndx>=elementNum||eIndx==0)return;

    baseIndx=vPSCounter[eIndx-1];
    if(vPSCounter[eIndx]==baseIndx)return;
    compactVector[2*baseIndx]=elementVector[2*eIndx];
    compactVector[2*baseIndx+1]=elementVector[2*eIndx+1];
    }
//==============================================================================


//=============================== Compact_CMF_Vector ================================
__global__ void Compact_CMF_Vector(long elementNum, long* vPSCounter, char* elementVector, char *compactVector){
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y;
    
    int blockIndx=bYIndx*gridDim.x+bXIndx;
    int eIndx=blockIndx*blockDim.x+tXIndx;
    int baseIndx=0;
    if(eIndx>=elementNum||eIndx==0)return;

    baseIndx=vPSCounter[eIndx-1];
    if(vPSCounter[eIndx]==baseIndx)return;
    compactVector[baseIndx]=elementVector[eIndx];
    }
//==============================================================================


//=============================== Compact_CMF_Vector ================================
__global__ void Make_Filtered_Index_List(long elementNum, long* vPSFlag, int *initIndexList, int *filteredList){
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y;
    
    int blockIndx=bYIndx*gridDim.x+bXIndx;
    int eIndx=blockIndx*blockDim.x+tXIndx;
    int baseIndx=0;
    if(eIndx>=elementNum)return;
    
    baseIndx=(eIndx==0?0:vPSFlag[eIndx-1]);
    if(vPSFlag[eIndx]==baseIndx)return;
    if(initIndexList!=NULL)eIndx=initIndexList[eIndx];
    filteredList[baseIndx]=eIndx;
    }
//==============================================================================


//=============================== CompactVector ================================
__global__ void CompactVector(long *elementNum, int* jCounter, int* prefixIndx, int* elementVector, int *compactVector, int* sideCrossCounter, int* bVNum, int* oVNum){
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y;
    
    int blockIndx=bYIndx*gridDim.x+bXIndx;
    int polIndx=blockIndx*blockDim.x+tXIndx;
    int baseIndx=0, bV;
    if(polIndx<*elementNum){
    	if(polIndx!=0)baseIndx=prefixIndx[polIndx-1];
        bV=bVNum[polIndx];
        for(int i=0;i<jCounter[polIndx];i++){
            compactVector[2*(baseIndx+i)]=polIndx;
            int oIndx=elementVector[(polIndx*GPU_MAX_CROSS_JOIN_PER_BASE+i)];
            compactVector[2*(baseIndx+i)+1]=oIndx;
            sideCrossCounter[baseIndx+i]=bV*oVNum[oIndx];
        }
    }
    }
//==============================================================================

//============================== CompactVector2 ================================
__global__ void CompactVector2(int baseRepeat, long* elementNum, int* jCounter, int* prefixIndx, int* elementVector, int *compactVector, int* sideCrossCounter, int* bVNum, int* oVNum){
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y, blockIndx, bV=0, baseIndx=0, polIndx, oIndx;
    long bucketIndx;
    blockIndx=bYIndx*gridDim.x+bXIndx;
    bucketIndx=blockIndx*blockDim.x+tXIndx;
    polIndx=bucketIndx/baseRepeat;
    if(polIndx<*elementNum){
        if(bucketIndx!=0)baseIndx=prefixIndx[bucketIndx-1];
        bV=bVNum[polIndx];
        for(int i=0;i<jCounter[bucketIndx];i++){
            compactVector[2*(baseIndx+i)]=polIndx;
            oIndx=elementVector[bucketIndx*GPU_MAX_CROSS_JOIN+i];
            compactVector[2*(baseIndx+i)+1]=oIndx;
            sideCrossCounter[baseIndx+i]=bV*oVNum[oIndx];
        }
    }
}
//==============================================================================

//============================== CompactVector3 ================================
__global__ void CompactVector3(int baseRepeat, long* elementNum, int* jCounter, int* prefixIndx, int* elementVector, int *compactVector, int* sideCrossCounter, int* bVNum, int* oVNum, float* bMBR, float* oMBR, float* cMBR){
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y, blockIndx, bV=0, baseIndx=0, polIndx, oIndx;
    long bucketIndx;
    blockIndx=bYIndx*gridDim.x+bXIndx;
    bucketIndx=blockIndx*blockDim.x+tXIndx;
    polIndx=bucketIndx/baseRepeat;
    if(polIndx<*elementNum){
        if(bucketIndx!=0)baseIndx=prefixIndx[bucketIndx-1];
        bV=bVNum[polIndx];
        float a1, a2, b1, b2, x1, x2, y1, y2, cX1, cY1, cX2, cY2;
        for(int i=0;i<jCounter[bucketIndx];i++){
           compactVector[2*(baseIndx+i)]=polIndx;
           oIndx=elementVector[bucketIndx*GPU_MAX_CROSS_JOIN+i];
           compactVector[2*(baseIndx+i)+1]=oIndx;
           sideCrossCounter[baseIndx+i]=bV*oVNum[oIndx];

           a1=bMBR[4*polIndx];
           a2=bMBR[4*polIndx+2];
           x1=oMBR[4*oIndx];
           x2=oMBR[4*oIndx+2];
           b1=bMBR[4*polIndx+1];
           b2=bMBR[4*polIndx+3];
           y1=oMBR[4*oIndx+1];
           y2=oMBR[4*oIndx+3];
           cMBR[4*(baseIndx+i)]=(a1>x1?a1:x1);
           cMBR[4*(baseIndx+i)+2]=(a2<x2?a2:x2);
           cMBR[4*(baseIndx+i)+1]=(b1>y1?b1:y1);
           cMBR[4*(baseIndx+i)+3]=(b2<y2?b2:y2);
        }
    }
}
//==============================================================================


//============================== CompactVector4 ================================
__global__ void CompactVector4(int baseRepeat, long elementNum, int* jCounter, int* prefixIndx, int* elementVector, int *compactVector, int* sideCrossCounter, int* bVNum, int* oVNum, mbr_t* bMBR, mbr_t* oMBR, mbr_t* cMBR, int* bPolyEdgeCounter, int* oPolyEdgeCounter){
    int tXIndx=threadIdx.x, bXIndx=blockIdx.x, bYIndx=blockIdx.y, blockIndx, bV=0, baseIndx=0, polIndx, oIndx;
    long bucketIndx;
    blockIndx=bYIndx*gridDim.x+bXIndx;
    bucketIndx=blockIndx*blockDim.x+tXIndx;
    polIndx=bucketIndx/baseRepeat;
    if(polIndx<elementNum){
        if(bucketIndx!=0)baseIndx=prefixIndx[bucketIndx-1];
        bV=bVNum[polIndx];
        float a1, a2, b1, b2, x1, x2, y1, y2, cX1, cY1, cX2, cY2;
        for(int i=0;i<jCounter[bucketIndx];i++){
           compactVector[2*(baseIndx+i)]=polIndx;
           oIndx=elementVector[bucketIndx*GPU_MAX_CROSS_JOIN+i];
           compactVector[2*(baseIndx+i)+1]=oIndx;
           sideCrossCounter[baseIndx+i]=bV*oVNum[oIndx];

           bPolyEdgeCounter[baseIndx+i]=bV;
           oPolyEdgeCounter[baseIndx+i]=oVNum[oIndx];

           a1=bMBR[4*polIndx];
           a2=bMBR[4*polIndx+2];
           x1=oMBR[4*oIndx];
           x2=oMBR[4*oIndx+2];
           b1=bMBR[4*polIndx+1];
           b2=bMBR[4*polIndx+3];
           y1=oMBR[4*oIndx+1];
           y2=oMBR[4*oIndx+3];
           cMBR[4*(baseIndx+i)]=(a1>x1?a1:x1);
           cMBR[4*(baseIndx+i)+2]=(a2<x2?a2:x2);
           cMBR[4*(baseIndx+i)+1]=(b1>y1?b1:y1);
           cMBR[4*(baseIndx+i)+3]=(b2<y2?b2:y2);
        }
    }
}
//==============================================================================
