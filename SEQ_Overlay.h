#include "Constants.h"
#include "Data_Structures.h"
//#include "Data_Visualization.h"
#include "clipper.hpp"
#include <stdio.h>
#include <math.h>
//#include <mpi.h>
//#include <shapelib/shapefil.h>

#include "GH-CUDA/lib/polyclip.cpp"
#include "GH-CUDA/lib/readShapefile.cpp"

using namespace ClipperLib;

//================================ GetEuclideanDist ===================================
float GetEuclideanDist(float x1, float y1, float x2, float y2){
   //printf("\n%f\t%f\t%f\t%f\n", x1,x2,y1,y2);
   return( sqrt(pow(x1-x2,2)+pow((y1-y2),2))  );
}
//===================================================================================

//================================ PointInPolygon ===================================
long SEQ_PointInPolygonTest(coord_t *bCoords, coord_t* oCoords, long pairNum, long pipNum, int* djxyVector, int* djPiPIndexList, char* dPiPType, long* bVPSNum, long* oVPSNum, int* dPiPFlag, int* djoinFlag){
   long wNum=0;
   int *jxyVector, *jPiPIndexList, *piPFlag, *joinFlag;
   char* pipType;
   CopyFromGPU((void**)&jxyVector, djxyVector, 2*sizeof(int)*pairNum, 1);
   CopyFromGPU((void**)&jPiPIndexList, djPiPIndexList, sizeof(int)*pipNum, 1);
   CopyFromGPU((void**)&pipType, dPiPType, sizeof(char)*pairNum, 1);
   int bIndx, oIndx, mappedIndx, c, k, vNum;
   coord_t pX, pY, *polyCoord, X1, X2, Y1, Y2;
   long bShift=0, oShift=0;
   for(int i=0;i<pipNum;i++){
       mappedIndx=jPiPIndexList[i];
       bIndx=jxyVector[2*mappedIndx];
       oIndx=jxyVector[2*mappedIndx+1];
       if(bIndx!=0)bShift=bVPSNum[bIndx-1];
       if(oIndx!=0)oShift=oVPSNum[oIndx-1];
       if(pipType[mappedIndx]==1){
         pX=*(bCoords+2*bShift);
         pY=*(bCoords+2*bShift+1);
         vNum=oVPSNum[oIndx]-oShift;
	 polyCoord=oCoords+2*oShift;
       }
       else{
         pX=*(oCoords+2*oShift);
         pY=*(oCoords+2*oShift+1);
         vNum=bVPSNum[bIndx]-bShift;
	 polyCoord=bCoords+2*bShift;
       }
       c=0;
       k=vNum-1;
       for(int j=0;j<vNum;k=j++){
         X1=*(polyCoord+2*j);
         Y1=*(polyCoord+2*j+1);
         X2=*(polyCoord+2*k);
         Y2=*(polyCoord+2*k+1);
         if((Y1>pY)==(Y2>pY))continue;
         if(pX<(X2-X1)*(pY-Y1)/(Y2-Y1) + X1){c=!c;}
        } 
        if(c)wNum++;
   }
   return wNum;
  }
//==============================================================================

//================================ CoordToMBR ===================================
mbr_t CoordToMBR(char* ct, char type){
  mbr_t retVal=0;
  const char fracNum=12;
  int fCount=0, fDot=0;
  switch(type){
    case 0:
      retVal=atof(ct);
      break;
    case 1:
      //"float" to "long long"
      char buff[50];
      int i=0;
      while(*ct){
	if(*ct!='.'){
           if(fCount<fracNum){buff[i++]=*(ct);if(fDot)fCount++;}
           else break;
        }
	else fDot=1;
	ct++;
	}
      for(;fCount<fracNum;fCount++)buff[i++]='0';
      buff[i]='\0';
      retVal=atoll(buff);
      retVal+=9000000000000000;
      //printf("\n%s %ld\n", buff, retVal);
      break;
  }
  return(retVal);
}
//==============================================================================

//================================ getVertex ===================================
int getVertex(char* buff, char tkBuff[][50], coord_t* tk, int i,int n){
    int j=0;
    char token[n][100];    
    for(int k=0;k<n;k++){
        j=0;
        //while(buff[i]!='.'&&buff[i]!='b'&&buff[i]!='-'&&(buff[i]<'0'||buff[i]>'9')&&i<strlen(buff))i++;
        //while(buff[i]=='b'||buff[i]=='.'||buff[i]=='-'||(buff[i]<='9'&&buff[i]>='0'))token[k][j++]=buff[i++];
        while(buff[i]!='.'&&buff[i]!='-'&&(buff[i]<'0'||buff[i]>'9')&&i<strlen(buff))i++;
        while(buff[i]=='.'||buff[i]=='-'||(buff[i]<='9'&&buff[i]>='0'))token[k][j++]=buff[i++];
        token[k][j]='\0';
        tk[k]=atof(token[k]);
        strcpy(tkBuff[k], token[k]);
        //printf("\nfBuff=%.13f \t token=%s\n", tk[k], token[k]);
    }
    if(i>=strlen(buff)){i=-2;}
    return(++i);
}
//==============================================================================




/*
//============================ ParallelReadTextFormatPolygon =============================
int ParallelReadTextFormatPolygon(const char* fileName, int* bVNum, long* bVPSNum, mbr_t* seqMBR, coord_t* seqMBR2, coord_t* baseCoords, long* bVNumSum, char mbrType, int maxPoly){
    MPI_File mpiFid;
    MPI_Init(0, NULL);
    return pCounter;
}
//==============================================================================
*/



//============================ ReadTextFormatPolygon =============================
int ReadTextFormatPolygon(const char* fileName, int* bVNum, long* bVPSNum, mbr_t* seqMBR, coord_t* seqMBR2, coord_t* baseCoords, long* bVNumSum, char mbrType, int maxPoly){
    int fNum=0, maxV=0;
    FILE *fid1;
    char tkBuff[2][50];
    char* buff=(char*)malloc(sizeof(char)*MAX_BUFF);
    coord_t fBuff[2], x1, y1, x2, y2, polyBuff[MAX_VERTICES][2];
    fid1=fopen(fileName,"rt");    
    int pCounter=0, i;
    while(fgets(buff,MAX_BUFF,fid1)!=NULL && pCounter<maxPoly){        
        char cX1[20], cY1[20], cX2[20], cY2[20]; 
        i=0;
        i=getVertex(buff, tkBuff, fBuff,i,1);
        fNum=0;
        x1=1000000;
        y1=1000000;
        x2=-1000000;
        y2=-1000000;
        while(i!=-1){        
            i=getVertex(buff, tkBuff, fBuff,i,2);
            if(i==-1&&fBuff[0]==0&&fBuff[1]==0)break;
            polyBuff[fNum][0]=fBuff[0];
            polyBuff[fNum++][1]=fBuff[1]; 
            if(fBuff[0]<x1){x1=fBuff[0];strcpy(cX1, tkBuff[0]);}
            if(fBuff[1]<y1){y1=fBuff[1];strcpy(cY1, tkBuff[1]);}
            if(fBuff[0]>x2){x2=fBuff[0];strcpy(cX2, tkBuff[0]);}
            if(fBuff[1]>y2){y2=fBuff[1];strcpy(cY2, tkBuff[1]);}
        }
        seqMBR[pCounter*4]=CoordToMBR(cX1, mbrType);
        seqMBR[pCounter*4+1]=CoordToMBR(cY1, mbrType);
        seqMBR[pCounter*4+2]=CoordToMBR(cX2, mbrType);
        seqMBR[pCounter*4+3]=CoordToMBR(cY2, mbrType);
        seqMBR2[pCounter*4]=atof(cX1);
        seqMBR2[pCounter*4+1]=atof(cY1);
        seqMBR2[pCounter*4+2]=atof(cX2);
        seqMBR2[pCounter*4+3]=atof(cY2);

        if(seqMBR[pCounter*4]>seqMBR[pCounter*4+2])swapElements(seqMBR, pCounter*4, pCounter*4+2);
        else if(seqMBR[pCounter*4]==seqMBR[pCounter*4+2]){printf("\nMBR Error!\n");continue;}
        if(seqMBR[pCounter*4+1]>seqMBR[pCounter*4+3])swapElements(seqMBR, pCounter*4+1, pCounter*4+3);
        else if(seqMBR[pCounter*4+1]==seqMBR[pCounter*4+3]){printf("\nMBR Error!\n");continue;}

        if(seqMBR2[pCounter*4]>seqMBR2[pCounter*4+2])swapElements(seqMBR2, pCounter*4, pCounter*4+2);
        else if(seqMBR2[pCounter*4]==seqMBR2[pCounter*4+2]){printf("\nMBR Error!\n");continue;}
        if(seqMBR2[pCounter*4+1]>seqMBR2[pCounter*4+3])swapElements(seqMBR2, pCounter*4+1, pCounter*4+3);
        else if(seqMBR2[pCounter*4+1]==seqMBR2[pCounter*4+3]){printf("\nMBR Error!\n");continue;}

//printf("\n%d:\t%ld  %s\t%ld  %s\t%ld  %s\t%ld  %s", pCounter, seqMBR[pCounter*4], cX1, seqMBR[pCounter*4+1], cY1, seqMBR[pCounter*4+2], cX2, seqMBR[pCounter*4+3], cY2);
//printf("\n%s %s %s %s", cX1, cY1, cX2, cY2);
        fNum-=2;
        for(int k=0;k<fNum;k++){
            *(baseCoords+2*(*bVNumSum+k))=polyBuff[k+2][0];
            *(baseCoords+2*(*bVNumSum+k)+1)=polyBuff[k+2][1];
        }      
        *bVNumSum+=fNum;
        bVPSNum[pCounter]=*bVNumSum;
        bVNum[pCounter++]=fNum;
        if(fNum>maxV)maxV=fNum;
    }
    
    fclose(fid1);
    //printf("\nMaximum number of vertices in the base polygons is : %d\n",maxV);
    return pCounter;
}
//==============================================================================




//============================ ReadTextFormatPolygon2 =============================
int ReadTextFormatPolygon2(const char* fileName, int* bVNum, long* bVPSNum, mbr_t* seqMBR, coord_t* seqMBR2, coord_t* baseCoords, long* bVNumSum, char mbrType, int maxPoly){
    int fNum=0, maxV=0;
    FILE *fid1;
    char tkBuff[2][50];
    char* buff=(char*)malloc(sizeof(char)*MAX_BUFF);
    coord_t fBuff[2], x1, y1, x2, y2, polyBuff[MAX_VERTICES][2];
    fid1=fopen(fileName,"rt");    
    int pCounter=0, i;
    double edgeSum;
    while(fgets(buff,MAX_BUFF,fid1)!=NULL && pCounter<maxPoly){        
        char cX1[20], cY1[20], cX2[20], cY2[20]; 
        i=0;
        i=getVertex(buff, tkBuff, fBuff,i,1);
        fNum=0;
        x1=1000000;
        y1=1000000;
        x2=-1000000;
        y2=-1000000;
        edgeSum=0;
        while(i!=-1){        
            i=getVertex(buff, tkBuff, fBuff,i,2);
            if(i==-1&&fBuff[0]==0&&fBuff[1]==0)break;
            polyBuff[fNum][0]=fBuff[0];
            polyBuff[fNum++][1]=fBuff[1]; 
            if(fBuff[0]<x1){x1=fBuff[0];strcpy(cX1, tkBuff[0]);}
            if(fBuff[1]<y1){y1=fBuff[1];strcpy(cY1, tkBuff[1]);}
            if(fBuff[0]>x2){x2=fBuff[0];strcpy(cX2, tkBuff[0]);}
            if(fBuff[1]>y2){y2=fBuff[1];strcpy(cY2, tkBuff[1]);}
        }
        seqMBR[pCounter*4]=CoordToMBR(cX1, mbrType);
        seqMBR[pCounter*4+1]=CoordToMBR(cY1, mbrType);
        seqMBR[pCounter*4+2]=CoordToMBR(cX2, mbrType);
        seqMBR[pCounter*4+3]=CoordToMBR(cY2, mbrType);
        seqMBR2[pCounter*4]=atof(cX1);
        seqMBR2[pCounter*4+1]=atof(cY1);
        seqMBR2[pCounter*4+2]=atof(cX2);
        seqMBR2[pCounter*4+3]=atof(cY2);

        if(seqMBR[pCounter*4]>seqMBR[pCounter*4+2])swapElements(seqMBR, pCounter*4, pCounter*4+2);
        else if(seqMBR[pCounter*4]==seqMBR[pCounter*4+2]){printf("\nMBR Error!\n");continue;}
        if(seqMBR[pCounter*4+1]>seqMBR[pCounter*4+3])swapElements(seqMBR, pCounter*4+1, pCounter*4+3);
        else if(seqMBR[pCounter*4+1]==seqMBR[pCounter*4+3]){printf("\nMBR Error!\n");continue;}

        if(seqMBR2[pCounter*4]>seqMBR2[pCounter*4+2])swapElements(seqMBR2, pCounter*4, pCounter*4+2);
        else if(seqMBR2[pCounter*4]==seqMBR2[pCounter*4+2]){printf("\nMBR Error!\n");continue;}
        if(seqMBR2[pCounter*4+1]>seqMBR2[pCounter*4+3])swapElements(seqMBR2, pCounter*4+1, pCounter*4+3);
        else if(seqMBR2[pCounter*4+1]==seqMBR2[pCounter*4+3]){printf("\nMBR Error!\n");continue;}

//printf("\n%d:\t%ld  %s\t%ld  %s\t%ld  %s\t%ld  %s", pCounter, seqMBR[pCounter*4], cX1, seqMBR[pCounter*4+1], cY1, seqMBR[pCounter*4+2], cX2, seqMBR[pCounter*4+3], cY2);
//printf("\n%s %s %s %s", cX1, cY1, cX2, cY2);
        fNum-=2;
        for(int k=0;k<fNum;k++){
            *(baseCoords+2*(*bVNumSum+k))=polyBuff[k+2][0];
            *(baseCoords+2*(*bVNumSum+k)+1)=polyBuff[k+2][1];
            if(k!=0)edgeSum+=GetEuclideanDist(*(baseCoords+2*(*bVNumSum+k-1)), *(baseCoords+2*(*bVNumSum+k-1)+1), *(baseCoords+2*(*bVNumSum+k)), *(baseCoords+2*(*bVNumSum+k)+1));
        }      
        *bVNumSum+=fNum;



        //printf("\nPoly, %d , Edges:, %d,  AverageEdgeLength:, %f, (w,h),  %f , %f", pCounter, fNum, bEdgeLen[pCounter], seqMBR2[pCounter*4+2]-seqMBR2[pCounter*4], seqMBR2[pCounter*4+3]-seqMBR2[pCounter*4+1]); 



        bVPSNum[pCounter]=*bVNumSum;
        bVNum[pCounter++]=fNum;


        if(fNum>maxV)maxV=fNum;
    }
    
    fclose(fid1);
    printf("\nMaximum number of vertices in this set of polygons is : %d\n",maxV);
    return pCounter;
}
//==============================================================================


//============================ ReadTextFormatPolygon2 with GH CPU data structure reading =============================
int ReadTextFormatPolygon2WithVector(const char* fileName, int* bVNum, long* bVPSNum, mbr_t* seqMBR, coord_t* seqMBR2, coord_t* baseCoords, long* bVNumSum, char mbrType, int maxPoly, vector<polygon>& PP){
    int fNum=0, maxV=0;
    FILE *fid1;
    char tkBuff[2][50];
    char* buff=(char*)malloc(sizeof(char)*MAX_BUFF);
    coord_t fBuff[2], x1, y1, x2, y2, polyBuff[MAX_VERTICES][2];
    fid1=fopen(fileName,"rt");    
    int pCounter=0, i;
    double edgeSum;

    point2D v;
    polygon P;

    while(fgets(buff,MAX_BUFF,fid1)!=NULL && pCounter<maxPoly){        
        char cX1[20], cY1[20], cX2[20], cY2[20]; 
        i=0;
        i=getVertex(buff, tkBuff, fBuff,i,1);
        fNum=0;
        x1=1000000;
        y1=1000000;
        x2=-1000000;
        y2=-1000000;
        edgeSum=0;
        while(i!=-1){        
            i=getVertex(buff, tkBuff, fBuff,i,2);
            if(i==-1&&fBuff[0]==0&&fBuff[1]==0)break;
            polyBuff[fNum][0]=fBuff[0];
            polyBuff[fNum++][1]=fBuff[1]; 
            if(fBuff[0]<x1){x1=fBuff[0];strcpy(cX1, tkBuff[0]);}
            if(fBuff[1]<y1){y1=fBuff[1];strcpy(cY1, tkBuff[1]);}
            if(fBuff[0]>x2){x2=fBuff[0];strcpy(cX2, tkBuff[0]);}
            if(fBuff[1]>y2){y2=fBuff[1];strcpy(cY2, tkBuff[1]);}
        }
        seqMBR[pCounter*4]=CoordToMBR(cX1, mbrType);
        seqMBR[pCounter*4+1]=CoordToMBR(cY1, mbrType);
        seqMBR[pCounter*4+2]=CoordToMBR(cX2, mbrType);
        seqMBR[pCounter*4+3]=CoordToMBR(cY2, mbrType);
        seqMBR2[pCounter*4]=atof(cX1);
        seqMBR2[pCounter*4+1]=atof(cY1);
        seqMBR2[pCounter*4+2]=atof(cX2);
        seqMBR2[pCounter*4+3]=atof(cY2);

        if(seqMBR[pCounter*4]>seqMBR[pCounter*4+2])swapElements(seqMBR, pCounter*4, pCounter*4+2);
        else if(seqMBR[pCounter*4]==seqMBR[pCounter*4+2]){printf("\nMBR Error!\n");continue;}
        if(seqMBR[pCounter*4+1]>seqMBR[pCounter*4+3])swapElements(seqMBR, pCounter*4+1, pCounter*4+3);
        else if(seqMBR[pCounter*4+1]==seqMBR[pCounter*4+3]){printf("\nMBR Error!\n");continue;}

        if(seqMBR2[pCounter*4]>seqMBR2[pCounter*4+2])swapElements(seqMBR2, pCounter*4, pCounter*4+2);
        else if(seqMBR2[pCounter*4]==seqMBR2[pCounter*4+2]){printf("\nMBR Error!\n");continue;}
        if(seqMBR2[pCounter*4+1]>seqMBR2[pCounter*4+3])swapElements(seqMBR2, pCounter*4+1, pCounter*4+3);
        else if(seqMBR2[pCounter*4+1]==seqMBR2[pCounter*4+3]){printf("\nMBR Error!\n");continue;}

//printf("\n%d:\t%ld  %s\t%ld  %s\t%ld  %s\t%ld  %s", pCounter, seqMBR[pCounter*4], cX1, seqMBR[pCounter*4+1], cY1, seqMBR[pCounter*4+2], cX2, seqMBR[pCounter*4+3], cY2);
//printf("\n%s %s %s %s", cX1, cY1, cX2, cY2);
        fNum-=2;
        for(int k=0;k<fNum;k++){
            *(baseCoords+2*(*bVNumSum+k))=polyBuff[k+2][0];
            *(baseCoords+2*(*bVNumSum+k)+1)=polyBuff[k+2][1];
            if(k!=0)edgeSum+=GetEuclideanDist(*(baseCoords+2*(*bVNumSum+k-1)), *(baseCoords+2*(*bVNumSum+k-1)+1), *(baseCoords+2*(*bVNumSum+k)), *(baseCoords+2*(*bVNumSum+k)+1));

            // ---------------------- FOR GH CPU data structure: start ----------------------
            v=point2D(polyBuff[k+2][0], polyBuff[k+2][1]);
			P.newVertex(v, true);
            // ---------------------- FOR GH CPU data structure: end ----------------------
        } 

        *bVNumSum+=fNum;
        // ---------------------- FOR GH CPU data structure: start ----------------------
        PP.push_back(P);
        P=polygon();
        // ---------------------- FOR GH CPU data structure: end ----------------------



        //printf("\nPoly, %d , Edges:, %d,  AverageEdgeLength:, %f, (w,h),  %f , %f", pCounter, fNum, bEdgeLen[pCounter], seqMBR2[pCounter*4+2]-seqMBR2[pCounter*4], seqMBR2[pCounter*4+3]-seqMBR2[pCounter*4+1]); 



        bVPSNum[pCounter]=*bVNumSum;
        bVNum[pCounter++]=fNum;


        if(fNum>maxV)maxV=fNum;
    }
    
    fclose(fid1);
    printf("\nMaximum number of vertices in this set of polygons is : %d\n",maxV);
    return pCounter;
}
//==============================================================================

 

//============================ ReadTextFormatPolygon3 =============================
int ReadTextFormatPolygon3(const char* fileName, int* bVNum, long* bVPSNum, mbr_t* seqMBR, coord_t* seqMBR2, coord_t* baseCoords, long* bVNumSum, char mbrType, int maxPoly){
    int fNum=0, maxV=0;
    FILE *fid1;
    char tkBuff[2][50];
    char* buff=(char*)malloc(sizeof(char)*MAX_BUFF);
    coord_t fBuff[2], x1, y1, x2, y2, polyBuff[MAX_VERTICES][2];
    fid1=fopen(fileName,"rt");
    int pCounter=0, i;
    double edgeSum;
    while(fgets(buff,MAX_BUFF,fid1)!=NULL && pCounter<maxPoly){        
        char cX1[20], cY1[20], cX2[20], cY2[20]; 
        i=0;
        //i=getVertex(buff, tkBuff, fBuff,i,1);
        fNum=0;
        x1=1000000;
        y1=1000000;
        x2=-1000000;
        y2=-1000000;
        edgeSum=0;
        while(i!=-1){        
            i=getVertex(buff, tkBuff, fBuff,i,2);
            if(i==-1&&fBuff[0]==0&&fBuff[1]==0)break;
            polyBuff[fNum][0]=fBuff[0];
            polyBuff[fNum++][1]=fBuff[1]; 
            if(fBuff[0]<x1){x1=fBuff[0];strcpy(cX1, tkBuff[0]);}
            if(fBuff[1]<y1){y1=fBuff[1];strcpy(cY1, tkBuff[1]);}
            if(fBuff[0]>x2){x2=fBuff[0];strcpy(cX2, tkBuff[0]);}
            if(fBuff[1]>y2){y2=fBuff[1];strcpy(cY2, tkBuff[1]);}
        }
        seqMBR[pCounter*4]=CoordToMBR(cX1, mbrType);
        seqMBR[pCounter*4+1]=CoordToMBR(cY1, mbrType);
        seqMBR[pCounter*4+2]=CoordToMBR(cX2, mbrType);
        seqMBR[pCounter*4+3]=CoordToMBR(cY2, mbrType);
        seqMBR2[pCounter*4]=atof(cX1);
        seqMBR2[pCounter*4+1]=atof(cY1);
        seqMBR2[pCounter*4+2]=atof(cX2);
        seqMBR2[pCounter*4+3]=atof(cY2);

        if(seqMBR[pCounter*4]>seqMBR[pCounter*4+2])swapElements(seqMBR, pCounter*4, pCounter*4+2);
        else if(seqMBR[pCounter*4]==seqMBR[pCounter*4+2]){
		//printf("\nMBR Error!\n");
		continue;}
        if(seqMBR[pCounter*4+1]>seqMBR[pCounter*4+3])swapElements(seqMBR, pCounter*4+1, pCounter*4+3);
        else if(seqMBR[pCounter*4+1]==seqMBR[pCounter*4+3]){
		//printf("\nMBR Error!\n");
		continue;}

        if(seqMBR2[pCounter*4]>seqMBR2[pCounter*4+2])swapElements(seqMBR2, pCounter*4, pCounter*4+2);
        else if(seqMBR2[pCounter*4]==seqMBR2[pCounter*4+2]){
		//printf("\nMBR Error!\n");
		continue;}
        if(seqMBR2[pCounter*4+1]>seqMBR2[pCounter*4+3])swapElements(seqMBR2, pCounter*4+1, pCounter*4+3);
        else if(seqMBR2[pCounter*4+1]==seqMBR2[pCounter*4+3]){
		//printf("\nMBR Error!\n");
		continue;}

//printf("\n%d:\t%ld  %s\t%ld  %s\t%ld  %s\t%ld  %s", pCounter, seqMBR[pCounter*4], cX1, seqMBR[pCounter*4+1], cY1, seqMBR[pCounter*4+2], cX2, seqMBR[pCounter*4+3], cY2);
//printf("\n%s \t %s \t %s \t %s\n", cX1, cY1, cX2, cY2);
//exit(0);
        fNum-=2;
        for(int k=0;k<fNum;k++){
            *(baseCoords+2*(*bVNumSum+k))=polyBuff[k+2][0];
            *(baseCoords+2*(*bVNumSum+k)+1)=polyBuff[k+2][1];
            if(k!=0)edgeSum+=GetEuclideanDist(*(baseCoords+2*(*bVNumSum+k-1)), *(baseCoords+2*(*bVNumSum+k-1)+1), *(baseCoords+2*(*bVNumSum+k)), *(baseCoords+2*(*bVNumSum+k)+1));
        }      
        *bVNumSum+=fNum;



        //printf("\nPoly, %d , Edges:, %d,  AverageEdgeLength:, %f, (w,h),  %f , %f", pCounter, fNum, bEdgeLen[pCounter], seqMBR2[pCounter*4+2]-seqMBR2[pCounter*4], seqMBR2[pCounter*4+3]-seqMBR2[pCounter*4+1]); 



        bVPSNum[pCounter]=*bVNumSum;
        bVNum[pCounter++]=fNum;


        if(fNum>maxV)maxV=fNum;
    }
    
    fclose(fid1);
    printf("\nMaximum number of vertices in this set of polygons is : %d\n",maxV);
    return pCounter;
}
//==============================================================================


//============================= IntersectMBR_SEQ ===============================
void SEQMBROverlap(long bPolNum, long oPolNum, mbr_t* seqMBR, mbr_t* seqOMBR, int mbrIndx){
    int counter=0, tempVector[5000];        

    mbr_t x1,x2,y1,y2,a1,a2,b1,b2;   
    for(int i=0;i<bPolNum;i++){
        a1=seqMBR[i*4];
        b1=seqMBR[i*4+1];
        a2=seqMBR[i*4+2];
        b2=seqMBR[i*4+3];
        for(int j=0;j<oPolNum;j++){
            x1=seqOMBR[j*4];
            y1=seqOMBR[j*4+1];
            x2=seqOMBR[j*4+2];
            y2=seqOMBR[j*4+3];            
            if(x2<=a1 || a2<=x1)continue;
            if(y2<=b1 || b2<=y1)continue; 
            
            if(i==mbrIndx || mbrIndx==-1)printf("\n(%d,%d)\t(%ld %ld %ld %ld)\t(%ld %ld %ld %ld)", i, j, a1, b1, a2, b2, x1, y1, x2, y2);
            //counter++;            
        }        
    }
    return;
}
//==============================================================================


//============================= IntersectMBR_SEQ ===============================
void IntersectMBR_SEQ(long bPolNum, long oPolNum, mbr_t* seqMBR, mbr_t* seqOMBR, int* jSeqVector[], int* jSeqCounter){
    int maxJ=0, maxJ2=0, maxJ3=0, counter=0, tempVector[5000];        

    mbr_t x1,x2,y1,y2,a1,a2,b1,b2;   
    for(int i=0;i<bPolNum;i++){
        a1=seqMBR[i*4];
        b1=seqMBR[i*4+1];
        a2=seqMBR[i*4+2];
        b2=seqMBR[i*4+3];
        jSeqCounter[i]=0;        
        for(int j=0;j<oPolNum;j++){
            x1=seqOMBR[j*4];
            y1=seqOMBR[j*4+1];
            x2=seqOMBR[j*4+2];
            y2=seqOMBR[j*4+3];            
            if(x2<=a1 || a2<=x1)continue;
            if(y2<=b1 || b2<=y1)continue; 
            counter++;            
            if(jSeqCounter[i]<MAX_CROSS_JOINT){               
                tempVector[jSeqCounter[i]]=j;
                jSeqCounter[i]++;                  
            }
            else{
                printf("\nError! Poly %d has exceeded the maximum cross joints.\n",i);
                exit(0);
            }                        
        }        
        if(jSeqCounter[i]==0)continue;

        if(maxJ<jSeqCounter[i])
        {
            maxJ3=maxJ2;
            maxJ2=maxJ;
            maxJ=jSeqCounter[i];
        }
        else if(maxJ2<jSeqCounter[i])
        {
            maxJ3=maxJ2;
            maxJ2=jSeqCounter[i];
        }
        else if(maxJ3<jSeqCounter[i])
        {
            maxJ3=jSeqCounter[i];
        }

        //printf("Sequential: Number of joint pairs of overlay MBRs with base MBR %d is: %d\n",i,jSeqCounter[i]);
        jSeqVector[i]=(int*)malloc(sizeof(int)*jSeqCounter[i]);
        for(int j=0;j<jSeqCounter[i];j++){
            *(jSeqVector[i]+j)=tempVector[j];
        //    printf("\t%d", tempVector[j]);
        }
        //printf("\n");        
    }
    //printf("\nMax number of joints: (1th: %d) \t (2th: %d) \t (3th: %d) of : %d\n", maxJ, maxJ2, maxJ3, counter);
    return;
}

//==============================================================================

//==============================================================================
void PrintLinkListPolygon(Vertex* vPoly){
    int cnt=0;
    while(1){
        if(vPoly==NULL)break;
        printf("%d : (%f , %f)\n",++cnt, vPoly->x,vPoly->y);
        vPoly=vPoly->next;
    }
    return;
}
//==============================================================================

//=============================== IntersectEdge ================================
bool IntersectEdge(mbr_t a1, mbr_t b1, mbr_t a2, mbr_t b2, mbr_t c1, mbr_t d1, mbr_t c2, mbr_t d2, mbr_t* x0, mbr_t* y0){
    mbr_t m1, m2;
    m1=(b2-b1)/(a2-a1);
    m2=(d2-d1)/(c2-c1);
    if(m1!=m2){
        *x0=(d1-b1+m1*a1-m2*c1)/(m1-m2);
        *y0=m1*(*x0-a1)+b1;
    }
    else{
        //printf("\nm1(%f)====m2(%f)\n",m1,m2);
        return false;        
    }
    if(a1<=*x0 && *x0<=a2 && c1<=*x0 && *x0<=c2 && b1<=*y0 && *y0<=b2 && d1<=*y0 && *y0<=d2){        
        return true;
    }
    else
        return false;
}
//==============================================================================

//============================== PointInPolyTest ===============================
bool PointInPolyTest(int vNum, mbr_t *Coords, mbr_t testx, mbr_t testy)
{
  bool c = false;
  for (int i = 0, j = vNum-1; i < vNum; j = i++) {
    if ( ((*(Coords+2*i+1)>testy) != (*(Coords+2*j+1)>testy)) &&
     (testx < (*(Coords+2*j)-*(Coords+2*i)) * (testy-*(Coords+2*i+1)) / (*(Coords+2*j+1)-*(Coords+2*i+1)) + *(Coords+2*i)) )
       c = !c;
  }
  return c;
}
//==============================================================================

//=============================== CreateVertex =================================
void CreateVertex(Vertex* bV0, Vertex* oV0, mbr_t x1, mbr_t y1){
    mbr_t bAlpha, oAlpha;
    bAlpha=(bV0->x-x1)*(bV0->x-x1)+(bV0->y-y1)*(bV0->y-y1);
    oAlpha=(oV0->x-x1)*(oV0->x-x1)+(oV0->y-y1)*(oV0->y-y1);
    Vertex *bV1, *oV1;    
    bV1=bV0;
    Vertex* newbV, *newoV;
    while(1){
        if(bV1->next==NULL || !bV1->next->intersect || bV1->next->alpha>=bAlpha){
            newbV=(Vertex*)malloc(sizeof(Vertex));
            newbV->x=x1;
            newbV->y=y1;
            if(bV1->next != NULL){
                bV1->next->previous=newbV;
            }            
            newbV->previous=bV1;
            newbV->next=bV1->next;
            bV1->next=newbV;  
            newbV->intersect=true;
            newbV->alpha=bAlpha;
            break;
        }
        else{
            bV1=bV1->next;
        }
    }
    
    oV1=oV0;
    while(1){
        if(oV1->next==NULL || !oV1->next->intersect || oV1->next->alpha>=oAlpha){
            newoV=(Vertex*)malloc(sizeof(Vertex));
            newoV->x=x1;
            newoV->y=y1;
            newoV->intersect=true;
            if(oV1->next != NULL){
                oV1->next->previous=newoV;
            }            
            newoV->previous=oV1;
            newoV->next=oV1->next;
            oV1->next=newoV;   
            newoV->alpha=oAlpha;
            break;
        }
        else{
            oV1=oV1->next;
        }
    }
    newbV->neighbor=newoV;
    newoV->neighbor=newbV;
}
//==============================================================================

//============================ IntersectEdges_SEQ ==============================
void IntersectEdges_SEQ(Vertex *bVPoly, Vertex *oVPoly, mbr_t* bCoords, mbr_t* oCoords, int bVNum, int oVNum, int bIndx, int oIndx){    
    mbr_t a1, a2, b1, b2, c1, c2, d1, d2,x0,y0;
    Vertex *bV, *oV;
    bV=bVPoly;    
    for(int i=0;i<bVNum;i++){        
        a1=*(bCoords+2*i);
        b1=*(bCoords+2*i+1);
        if(i<bVNum-1){
            a2=*(bCoords+2*(i+1));
            b2=*(bCoords+2*(i+1)+1);
        }
        else{
            a2=*(bCoords);
            b2=*(bCoords+1);            
        }
        oV=oVPoly;
        for(int j=0;j<oVNum;j++){
            c1=*(oCoords+2*j);
            d1=*(oCoords+2*j+1);
            if(j<oVNum-1){
                c2=*(oCoords+2*(j+1));
                d2=*(oCoords+2*(j+1)+1);
            }
            else{
                c2=*(oCoords);
                d2=*(oCoords+1);            
            }
            if(IntersectEdge(a1,b1,a2,b2,c1,d1,c2,d2,&x0,&y0)){                
                /*
                PrintLinkListPolygon(bVPoly);
                PrintLinkListPolygon(oVPoly);
                printf("\n(%d , %d)::(%f , %f)\tB : %d ,([%f,%f]-[%f,%f])\t O : %d ,([%f,%f]-[%f,%f])\n",i,j,x0,y0,bVNum,a1,b1,a2,b2,oVNum,c1,d1,c2,d2);
                */
                CreateVertex(bV, oV, x0, y0);
                //PrintLinkListPolygon(bVPoly);
                //PrintLinkListPolygon(oVPoly);
                //exit(0);                
            }
            if(oV->next != NULL)oV=oV->next;   
        }
        if(bV->next != NULL)bV=bV->next;        
    }
}
//==============================================================================

//=========================== CreateLinkListPolygon ============================
Vertex* CreateLinkListPolygon(mbr_t* coords, int vNum){
    Vertex *vPoly0, *vPoly1, *vPoly2;
    vPoly0=(Vertex*)malloc(sizeof(Vertex));
    vPoly0->x=*(coords);
    vPoly0->y=*(coords+1);
    vPoly0->previous=NULL;
    vPoly1=vPoly0;
    for(int i=1;i<vNum;i++){
        vPoly2=(Vertex*)malloc(sizeof(Vertex));
        vPoly2->x=*(coords+2*i);
        vPoly2->y=*(coords+2*i+1);
        vPoly2->previous=vPoly1;
        vPoly1->next=vPoly2;
        vPoly1=vPoly2;
    }
    vPoly2->next=NULL;
    return vPoly0;
}
//==============================================================================

//=========================== UpdateEntryExitStatus ============================
void UpdateEntryExitStatus(Vertex *mainPoly, Vertex *otherPoly, mbr_t* otherCoords, mbr_t otherVNum){
    mbr_t xP0=mainPoly->x,yP0=mainPoly->y;
    bool pipTest=!(PointInPolyTest(otherVNum, otherCoords, xP0,yP0));
    Vertex *mV1;    
    mV1=mainPoly;
    while(mV1!=NULL){        
        if(mV1->intersect){
            mV1->entry_exit=pipTest;
            pipTest=!pipTest;
        }            
        mV1=mV1->next;
    }
}
//==============================================================================

//=========================== IntersectPolygons_SEQ ============================
void IntersectPolygons_SEQ(long bPolNum, long oPolNum, mbr_t* bCoords[], mbr_t* oCoords[], int * bVNum, int *oVNum, int* jSeqVector[], int* jSeqCounter){    
    Vertex* bVPoly, *oVPoly;
    for(int i=0;i<bPolNum;i++){
        for(int j=0;j<jSeqCounter[i];j++){            
            int k=*(jSeqVector[i]+j);            
            bVPoly=CreateLinkListPolygon(bCoords[i], bVNum[i]);
            oVPoly=CreateLinkListPolygon(oCoords[k], oVNum[k]);            
            IntersectEdges_SEQ(bVPoly, oVPoly, bCoords[i], oCoords[k], bVNum[i], oVNum[k],i,k);
            UpdateEntryExitStatus(bVPoly, oVPoly, oCoords[k], oVNum[k]);
            UpdateEntryExitStatus(oVPoly, bVPoly, bCoords[i], bVNum[i]);                        
            free(bVPoly);
            bVPoly=NULL;
            free(oVPoly);
            oVPoly=NULL;
            //PrintLinkListPolygon(bVPoly);
            //PrintLinkListPolygon(oVPoly);
            //exit(0);            
            
        }
    }
}
//==============================================================================

//============================ CreatePolygonObject =============================
void CreatePolygonObject(mbr_t* coords, int vNum, Path *polObj){
    mbr_t x,y;
    for(int i=0;i<vNum;i++){
        x=coords[2*i];
        y=coords[2*i+1];
        polObj->insert(polObj->begin()+i, IntPoint(x,y));
    }
    return;
}
//==============================================================================
