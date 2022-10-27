#include "Constants.h"
#include "Types.h"
#include <stdio.h>

#ifdef DEFINE_VARIABLES
#define EXTERN 
#else
#define EXTERN extern
#endif 

//======================= PrintPairs ========================
void PrintPairs(int* gpuVector, int* gpuFlag, long pairNum){
   int *vector, *flag; 
   CopyFromGPU((void**)&vector, gpuVector, 2*sizeof(int)*pairNum, 1);
   CopyFromGPU((void**)&flag, gpuFlag, sizeof(int)*pairNum, 1);
   for(int i=0;i<pairNum;i++)if(flag[i]==1)printf("\n(%d,%d)", vector[2*i], vector[2*i+1]);
   free(flag);
   free(vector);
   return;
}
//==============================================================================


//======================= PrintPolygon ========================
void PrintPolygon(coord_t* polyCoord, int pNum){
    printf("Polygon size %d\n", pNum);
   for(int i=0;i<pNum;i++){
     printf("%.13f %.13f\n", polyCoord[i*2], polyCoord[i*2+1]);
   }
   return;
}

//======================= PrintIntersectedMBRVector_SEQ ========================
void ValidateMBRs(float *seqMBR, float* gpuMBR, int polNum){
    for(int i=0;i<polNum;i++){
        //printf("\nMBR %d : seq:(%f , %f , %f , %f)\tGPU:(%f , %f , %f , %f)",i,seqMBR[i*4],seqMBR[i*4+1],seqMBR[i*4+2],seqMBR[i*4+3],gpuMBR[i*4],gpuMBR[i*4+1],gpuMBR[i*4+2],gpuMBR[i*4+3]);
        if(gpuMBR[i*4]!=seqMBR[i*4] || gpuMBR[i*4+1]!=seqMBR[i*4+1] || gpuMBR[i*4+2]!=seqMBR[i*4+2] || gpuMBR[i*4+3]!=seqMBR[i*4+3]){
            printf("\nERROR: Inconsistency in MBR %d : seq:(%f , %f , %f , %f)\tGPU:(%f , %f , %f , %f)\n",i,seqMBR[i*4],seqMBR[i*4+1],seqMBR[i*4+2],seqMBR[i*4+3],gpuMBR[i*4],gpuMBR[i*4+1],gpuMBR[i*4+2],gpuMBR[i*4+3]);
            //exit(1);
        }
    }
    printf("\nMBRs are fully consistent.\n");
    return;
}
//==============================================================================


//======================= PrintIntersectedMBRVector_SEQ ========================
void ValidateCoordinations2(float * seqCoords, float *gpuCoords, int *seqVNum, int *gpuVNum, int seqPolNum, int gpuPolNum){
    if(seqPolNum!=gpuPolNum){
        printf("\n\t Error: Inconsistency in number of polygons \n");
        exit(1);
    }
    int polBaseShift=0;
    for(int i=0;i<gpuPolNum;i++){
        if(seqVNum[i]!=gpuVNum[i]){
            printf("\n\t Error: Inconsistency in number of vertices in polygon number %d \n",i);
            exit(1);            
        }
        for(int j=0;j<seqVNum[i];j++){
            if(*(seqCoords+polBaseShift+2*j)!=gpuCoords[polBaseShift+j*2] || *(seqCoords+polBaseShift+2*j+1)!=gpuCoords[polBaseShift+j*2+1]){
                printf("\n\nError: Inconsistency in polygon number %d coordination index %d  : seq:(%f,%f) VS gpu:(%f,%f)\n\n", i, j, *(seqCoords+polBaseShift+2*j), *(seqCoords+polBaseShift+2*j+1), gpuCoords[polBaseShift+j*2], gpuCoords[polBaseShift+j*2+1]);
                exit(1);
            }
        }
        polBaseShift+=2*seqVNum[i];
    }
    printf("\nCoordinations are fully consistent.\n");
    return;
}
//==============================================================================


//======================= PrintIntersectedMBRVector_SEQ ========================
void ValidateCoordinations(float * seqCoords[], float *gpuCoords, int *seqVNum, int *gpuVNum, int seqPolNum, int gpuPolNum){
    if(seqPolNum!=gpuPolNum){
        printf("\n\t Error: Inconsistency in number of polygons \n");
        exit(1);
    }
    int polBaseShift=0;
    for(int i=0;i<gpuPolNum;i++){
        if(seqVNum[i]!=gpuVNum[i]){
            printf("\n\t Error: Inconsistency in number of vertices in polygon number %d \n",i);
            exit(1);            
        }
        for(int j=0;j<seqVNum[i];j++){
            if(*(seqCoords[i]+2*j)!=gpuCoords[polBaseShift+j*2] || *(seqCoords[i]+2*j+1)!=gpuCoords[polBaseShift+j*2+1]){
                printf("\n\nError: Inconsistency in polygon number %d coordination index %d  : seq:(%f,%f) VS gpu:(%f,%f)\n\n", i, j, *(seqCoords[i]+2*j), *(seqCoords[i]+2*j+1), gpuCoords[polBaseShift+j*2], gpuCoords[polBaseShift+j*2+1]);
                exit(1);
            }
        }
        polBaseShift+=2*seqVNum[i];
    }
    printf("\nCoordinations are fully consistent.\n");
    return;
}
//==============================================================================

//======================= PrintIntersectedMBRVector_SEQ ========================
void PrintIntersectedMBRVector_SEQ(int* jVector[], int* jCounter, int bPolNum){
    for(int i=0;i<bPolNum;i++){        
        if(jCounter[i]<=0)continue;
        printf("\n");
        printf("\nBase polygon's MBR %d has overlap with: ", i);        
        for(int j=0;j<jCounter[i];j++){
            printf("%d, ",*(jVector[i]+j));
        }
    }
}
//==============================================================================

//=============== PrintIntersectedMBRVector_GPU ==================
void PrintIntersectedMBRVector_BucketIntegrated_GPU(int* jVector, int* jCounter, int bPolNum, int oPolNum){
    int baseRepeat=(oPolNum/oBucketLength), flag, indx;
    if(baseRepeat*oBucketLength<oPolNum)baseRepeat++;
    for(int i=0;i<bPolNum;i++){
        flag=0;
        for(int j=0;j<baseRepeat;j++){
            indx=(i)*baseRepeat+j;
            if(jCounter[indx]<=0)continue;
            if(!flag){
                flag=1;
                printf("\n\nBase polygon's MBR %d has overlap with: ", i);        
            }
            for(int k=0;k<jCounter[indx];k++){
               // if(jVector[indx*GPU_MAX_CROSS_JOIN+k]!=1024)continue;
                printf("%d\t",jVector[indx*GPU_MAX_CROSS_JOIN+k]);
            }
        }
    }
}
//==============================================================================

//=============== PrintIntersectedMBRVector_BucketSplited_GPU ==================
void PrintIntersectedMBRVector_BucketSplited_GPU(int* jVector, int* jCounter, int bPolNum, int oPolNum){
    int baseRepeat=(oPolNum/oBucketLength);
    for(int i=0;i<bPolNum;i++){
        for(int j=0;j<baseRepeat;j++){
            int indx=(i)*baseRepeat+j;
            if(jCounter[indx]<=0)continue;
            printf("Base MBR %d Part %d # of Joins: %d\n",i,j , jCounter[indx]);
            for(int k=0;k<jCounter[indx];k++){
                printf("\t%d",jVector[indx*GPU_MAX_CROSS_JOIN+k]);
                //if(indx*GPU_MAX_CROSS_JOIN+k==4896)printf("\J4896: \t%d\n",jVector[4896]);
            }
            printf("\n");            
        }
    }
}
//==============================================================================

//=============== PrintIntersectedMBRVector_Compact_GPU ==================
void PrintIntersectedMBRVector_Compact_GPU(int* jCompactVector, int* jCounter, int* jPSCounter, int bPolNum, int baseRepeat){
    int baseIndx=0;
    for(int i=0;i<bPolNum*baseRepeat;i++){
        if(i>0)baseIndx=jPSCounter[i-1];
        if(jCounter[i]<=0)continue;
        printf("\n\nBase polygon's MBR %d has overlap with: ", i/baseRepeat);        
        for(int k=0;k<jCounter[i];k++){
            printf("%d,%d\t",jCompactVector[2*(baseIndx+k)],jCompactVector[2*(baseIndx+k)+1]);
        }
    }    
}
//==============================================================================

//=============== PrintIntersectedMBRVector_BucketSplited_GPU ==================
void PrintBucketPairVector(int* eVector, int* eCounter, int eNum, int bucketSize, int mbrIndx){
    int shift;
    for(int i=0;i<eNum;i++){       
	shift=i*bucketSize*2;
        if(eCounter[i]<=0)continue;
        //printf("MBR  %d , Number of Elements: %d\n",i , eCounter[i]);
        for(int k=0;k<eCounter[i];k++){

            if(mbrIndx==-1 || eVector[shift+2*k]!=mbrIndx)continue;
            printf("\t(%d , %d)\n",eVector[shift+2*k], eVector[shift+2*k+1]);
        }
        //printf("\n");            
    }
}
//==============================================================================

//=============== PrintIntersectedMBRRVector_BucketSplited_GPU ==================
void PrintBucketPairRVector(int* eVector, int* eCounter, int eNum, int bucketSize){
    int shift;
    for(int i=0;i<eNum;i++){       
	shift=i*bucketSize*2;
        if(eCounter[i]<=0)continue;
        //printf("MBR  %d , Number of Elements: %d\n",i , eCounter[i]);
        for(int k=0;k<eCounter[i];k++){

            if(eVector[shift+2*k+1]!=35)continue;
            printf("\t(%d , %d)\n",eVector[shift+2*k], eVector[shift+2*k+1]);
        }
        //printf("\n");            
    }
}
//==============================================================================

//================================= CountPairs =================================
void CountPairs(int* eVector, int* eCounter, int eNum, int bucketSize){
    int shift, pCounter[eNum], mIndx;
    for(int i=0;i<eNum;i++)pCounter[i]=0;       
    for(int i=0;i<eNum;i++){       
	shift=i*bucketSize*2;
        if(eCounter[i]<=0)continue;
        for(int k=0;k<eCounter[i];k++){
            mIndx=eVector[shift+2*k];
            //if(mIndx!=11)continue;
            //if(mIndx>10000)printf("\nERROR!\n");
            pCounter[mIndx]++;
        }
    }
    for(int i=0;i<eNum;i++)if(pCounter[i]>0)printf("\nMBR %d has %d pairs", i, pCounter[i]);            
}
//==============================================================================

//================================= RCountPairs =================================
void RCountPairs(int* eVector, int* eCounter, int eNum, int bucketSize){
    int shift, pCounter[eNum], mIndx;
    for(int i=0;i<eNum;i++)pCounter[i]=0;       
    for(int i=0;i<eNum;i++){       
	shift=i*bucketSize*2;
        if(eCounter[i]<=0)continue;
        for(int k=0;k<eCounter[i];k++){
            mIndx=eVector[shift+2*k+1];
            //if(mIndx!=11)continue;
            //if(mIndx>10000)printf("\nERROR!\n");
            pCounter[mIndx]++;
        }
    }
    for(int i=0;i<eNum;i++)if(pCounter[i]>0)printf("\nMBR %d has %d pairs", i, pCounter[i]);            
}
//==============================================================================

//========================== PrintPresumVectorTest =============================
void PrintPresumVectorTest(int* Vector, long* pSVector, long vNum, int printAll){
    long pFSum=0;
    for(long i=0;i<vNum;i++){
        pFSum+=Vector[i];
        if(Vector[i]==0)continue;
        if(printAll==0 && pFSum==pSVector[i])continue;
        printf("\nPrefix Sum up to index %ld of %ld is: (%d , %d):%d", i, vNum, Vector[i], pFSum, pSVector[i]);
        if(printAll==0)return;
    }
    printf("\nSum over all elements: %d\n", pFSum);
    printf("\nPrefix vector is successfully verified.\n");
}
//==============================================================================


//========================== PrintPresumVectorTest =============================
void PrintPresumVectorTest(int* Vector, int* pSVector, long vNum, int printAll){
    int pFSum=0;
    for(long i=0;i<vNum;i++){
        pFSum+=Vector[i];
        if(Vector[i]==0)continue;
        if(printAll==0 && pFSum==pSVector[i])continue;
        printf("\nPrefix Sum up to index %ld of %ld is: (%d , %d):%d", i, vNum, Vector[i], pFSum, pSVector[i]);
        if(printAll==0)return;
    }
    printf("\nSum over all elements: %d\n", pFSum);
    printf("\nPrefix vector is successfully verified.\n");
}
//==============================================================================

//============================ IndexCoverageTest ===============================
void IndexCoverageTest(long eNum, int* sortIndex){
   int coverageTest[eNum];    
   bool flag=false;
   for(int k=0;k<eNum;k++){
      coverageTest[k]=0;
   }
   for(int k=0;k<eNum;k++){
      coverageTest[sortIndex[k]]=1;
   }

   /*for(int k=0;k<eNum;k++){
      if(sortIndex[k]>=eNum){printf("\nInvalid range of %ld stored in sortIndex at %d\n",sortIndex[k], k);return;}
      if(coverageTest[sortIndex[k]]!=-1){
        printf("\n%d and %d hit same index[%d]\n",coverageTest[sortIndex[k]], k, sortIndex[k]);
        *k1=coverageTest[sortIndex[k]];
        *k2=k;
        //return;
      }
      coverageTest[sortIndex[k]]=k;
   }
   return;*/

   for(int k=0;k<eNum;k++){
      if(coverageTest[k]==0){printf("\nUncovered index at : %d\n", k);flag=true;return;}
      //if(coverageTest[k]>1){printf("\nIndex at : %d  covered %d times", k, coverageTest[k]);flag=true;}
   }
   if(!flag)printf("\n\n Full coverage for index range [0-%d]", eNum);
   return;
}
//==============================================================================


//============================ PrintSortedData ===============================
void PrintSortedData(int eNum, mbr_t* xData, mbr_t* yData, int* xSortIndex, int* ySortIndex){
   printf("\nPrinting out the sorted sequence...\n");
   for(int k=0;k<eNum;k++){
     //printf("\n%d:\t\tX[%d]:(%d)\tY[%d]:(%d)", k, xSortIndex[k], xData[xSortIndex[k]], ySortIndex[k], yData[ySortIndex[k]]);
      // if(xSortIndex[k]<10)
      //printf("\n%d:\t\tX[%d]:(%d)", k, xSortIndex[k], xData[xSortIndex[k]]);
      printf("\n%d:\t\tY[%d]:(%d)", k, ySortIndex[k], yData[ySortIndex[k]]);
   }
   return;
}
//==============================================================================


//============================ PrintSortedData ===============================
void PrintSortedData(int eNum, long long* xData, long long* yData, int* xPSCounter, int* yPSCounter, int* xSortIndex, int* ySortIndex, int digitPos){
   int sumX, sumY, sXIndx, sYIndx;
   sXIndx=eNum-1;
   sYIndx=eNum-1;
   sumX=0;
   sumY=0;
   printf("\n\n\nDigit buckets at position: %d", digitPos);
   for(int k=0;k<10;k++){
     sumX+=*(xPSCounter+(sXIndx)*10+k);
     sumY+=*(yPSCounter+(sYIndx)*10+k);
     printf("\nDigit bucket(%d): %d\t\tX: %d\t\tY: %d", eNum, k, sumX , sumY);
   }
   printf("\nPrinting out the sorted sequence...\n");
   for(int k=0;k<eNum;k++){
      printf("\n%d:\t\tX[%d]:(%d)\tY[%d]:(%d)", k, xSortIndex[k], xData[xSortIndex[k]], ySortIndex[k], yData[ySortIndex[k]]);
   }
   return;
}
//==============================================================================

//============================ PrintSortedData ===============================
void TestMBRIndices(int bPolNum, int oPolNum, cmbr_t* xMBR, cmbr_t* yMBR, int* xMBRIndex, int* yMBRIndex, int* xSortIndex, int* ySortIndex){
   int x1, x2, y1, y2;
   for(int k=0;k<bPolNum+oPolNum;k++){
      x1=xMBRIndex[2*k];
      x2=xMBRIndex[2*k+1];
      y1=yMBRIndex[2*k];
      y2=yMBRIndex[2*k+1];
      //printf("\nMBR %d: (x1:%d , y1:%d)-->[%d , %d]\t(x2:%d , y2:%d)-->[%d , %d]", k, x1, y1, xSortIndex[x1], ySortIndex[y1], x2, y2, xSortIndex[x2], ySortIndex[y2]);
      printf("\nX1(%d):%d=%d \t X2(%d):%d=%d \t Y1(%d):%d=%d \t Y2(%d):%d=%d", x1, 2*k, xSortIndex[x1], x2, 2*k+1, xSortIndex[x2], y1, 2*k, ySortIndex[y1], y2, 2*k+1 , ySortIndex[y2]);
   }
   return;
}
//==============================================================================

//============================ PrintDimensions ===============================
void PrintDimensions(int* jxbVector, int* jxbCounter, int* jxoVector, int* jxoCounter, int* jybVector, int* jybCounter, int* jyoVector, int* jyoCounter, long bPolNum, long oPolNum){
   int shift, sw=1, sum=0;
   printf("\n\n\nXB\n");
   for(int i=0;i<bPolNum;i++){
      if(sw!=1 && i!=0)continue;
      if(jxbCounter[i]==0)continue;
      //printf("\n\t%d has %d pairs:\n",i, jxbCounter[i]);
      sum+=jxbCounter[i];
      continue;
      for(int j=0;j<jxbCounter[i];j++){
         shift=2*GPU_MAX_JOIN_PER_DIM*i;
         printf("\t(%d , %d)", *(jxbVector+shift+2*j), *(jxbVector+shift+2*j+1));
      }
   }
   printf("\n\nNumber of total pairs from XB: %d\n", sum);

   printf("\n\n\nXO\n");
   sum=0;
   for(int i=0;i<oPolNum;i++){
      if(sw!=1 && i!=0)continue;
      if(jxoCounter[i]==0)continue;
      //printf("\n\t%d has %d pairs:\n",i, jxoCounter[i]);
      sum+=jxoCounter[i];
      continue;
      for(int j=0;j<jxoCounter[i];j++){
         shift=2*GPU_MAX_JOIN_PER_DIM*i;
         printf("\t(%d , %d)", *(jxoVector+shift+2*j), *(jxoVector+shift+2*j+1));
      }
   }
   printf("\n\nNumber of total pairs from XO: %d\n", sum);

   printf("\n\n\nYB\n");
   sum=0;
   for(int i=0;i<bPolNum;i++){
      if(sw!=1 && i!=0)continue;
      if(jybCounter[i]==0)continue;
      //printf("\n\t%d has %d pairs:\n",i, jybCounter[i]);
      sum+=jybCounter[i];
      continue;
      for(int j=0;j<jybCounter[i];j++){
         shift=2*GPU_MAX_JOIN_PER_DIM*i;
         printf("\t(%d , %d)", *(jybVector+shift+2*j), *(jybVector+shift+2*j+1));
      }
   }
   printf("\n\nNumber of total pairs from YB: %d\n", sum);

   printf("\n\n\nYO\n");
   sum=0;
   for(int i=0;i<oPolNum;i++){
      if(sw!=1 && i!=0)continue;
      if(jyoCounter[i]==0)continue;
      //printf("\n\t%d has %d pairs:\n",i, jyoCounter[i]);
      sum+=jyoCounter[i];
      continue;
      for(int j=0;j<jyoCounter[i];j++){
         shift=2*GPU_MAX_JOIN_PER_DIM*i;
         printf("\t(%d , %d)", *(jyoVector+shift+2*j), *(jyoVector+shift+2*j+1));
      }
   }
   printf("\n\nNumber of total pairs from YO: %d\n", sum);
   return;
}

//==============================================================================


//============================ PrintDimensions ===============================
void PrintDimensions(int* jxoVector, int* jxoCounter, int* jyoVector, int* jyoCounter, long oPolNum){
   int shift, sw=1, sum=0;

   printf("\n\n\nXO\n");
   sum=0;
   for(int i=0;i<oPolNum;i++){
      if(sw!=1)continue;
      if(jxoCounter[i]==0)continue;
      //printf("\n\t%d has %d pairs:\n",i, jxoCounter[i]);
      sum+=jxoCounter[i];
      continue;
      for(int j=0;j<jxoCounter[i];j++){
         shift=2*GPU_MAX_JOIN_PER_DIM*i;
         printf("\t(%d , %d)", *(jxoVector+shift+2*j), *(jxoVector+shift+2*j+1));
      }
   }
   printf("\n\nNumber of total pairs from XO: %d\n", sum);


   printf("\n\n\nYO\n");
   sum=0;
   for(int i=0;i<oPolNum;i++){
      if(sw!=1)continue;
      if(jyoCounter[i]==0)continue;
      //printf("\n\t%d has %d pairs:\n",i, jyoCounter[i]);
      sum+=jyoCounter[i];
      continue;
      for(int j=0;j<jyoCounter[i];j++){
         shift=2*GPU_MAX_JOIN_PER_DIM*i;
         printf("\t(%d , %d)", *(jyoVector+shift+2*j), *(jyoVector+shift+2*j+1));
      }
   }
   printf("\n\nNumber of total pairs from YO: %d\n", sum);
   return;
}
//==============================================================================


//============================ TestPSDigitCounter ===============================
void TestPSDigitCounter(int *xDigitCounter, int* xPSDigitCounter, int* xSortIndex, int elementNum, int i){
    int pSumTest;
    pSumTest=0;
    for(int i=0;i<10;i++){pSumTest+=*(xPSDigitCounter+(elementNum-1)*10+i);printf("%d:(%d , %d)\t", i, *(xPSDigitCounter+(elementNum-1)*10+i) , pSumTest);}
      pSumTest=0;
      int fflag;
      fflag=0;
      for(int j=0;j<elementNum;j++){
         pSumTest+=*(xDigitCounter+xSortIndex[j]*10*MAX_DIGITS+i*10+9);
      	 if(pSumTest!=*(xPSDigitCounter+j*10+9)) {fflag=1;printf("\n%d<>%d at %d\n",pSumTest, *(xPSDigitCounter+j*10+9), j);break;}
         else if(j==elementNum-1)printf("\nGOOD at Digit %d!\n\n", i);
     }
   return;
   }
//==============================================================================


//============================ TestPSDigitCounter ===============================
void PrintJoin(int* actualEdgeCrossCounter, int* jCompactVector, long pairNum){
   int cnt=0;
   for(int i=0;i<pairNum;i++){
     if(actualEdgeCrossCounter[i]==0)continue;
     printf("\n%d:\t(%d , %d) polygon pairs with %d edge intersections\n", ++cnt, jCompactVector[i*2], jCompactVector[i*2+1], actualEdgeCrossCounter[i]);
   }
}
//==============================================================================

