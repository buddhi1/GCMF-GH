#include "Constants.h"


void swapElements(mbr_t* data, int i, int j){
  mbr_t temp=data[i];
  data[i]=data[j];
  data[j]=temp;
  return;
}
void swapElements(coord_t* data, int i, int j){
  coord_t temp=data[i];
  data[i]=data[j];
  data[j]=temp;
  return;
}
//=========================== ReadSecondLayerData ==============================
long ReadSynthesizedMBRs(const char* fileName, mbr_t* seqMBR, long MaxMBRNum){
    FILE *fid;
    mbr_t xAvg=0, yAvg=0;
    long pCounter=0;
    char dummyBuff[20], buff[200], numBuff[20];
    fid=fopen(fileName,"rt");        
    if(fid==NULL){printf("File \"%s\" cannot be opened.", fileName);return(-1);} 
    while(pCounter<MaxMBRNum && fgets(buff, 200, fid)!=NULL){        
        int indx, nIndx;
        indx=0;
        while(buff[indx]!=':')indx++;
        indx++;
        for(int i=0;i<4;i++){
           while(buff[indx]==' ' || buff[indx]=='\t')indx++;
           nIndx=0;
           while(buff[indx]!='\0' && buff[indx]!=' ' && buff[indx]!='\t '&& buff[indx]!='\n')numBuff[nIndx++]=buff[indx++];
           numBuff[nIndx]='\0';
           seqMBR[pCounter*4+i]=atol(numBuff);
           //if(pCounter==0)printf("\n%ld  ,  %s \t %d", seqMBR[pCounter*4+i], numBuff, indx);
	} 
        if(seqMBR[pCounter*4]==0 || seqMBR[pCounter*4+1]==0 || seqMBR[pCounter*4+2]==0 || seqMBR[pCounter*4+3]==0)continue;
        if(seqMBR[pCounter*4]>seqMBR[pCounter*4+2])swapElements(seqMBR, pCounter*4, pCounter*4+2);
        if(seqMBR[pCounter*4+1]>seqMBR[pCounter*4+3])swapElements(seqMBR, pCounter*4+1, pCounter*4+3);
        xAvg+=seqMBR[pCounter*4+2]-seqMBR[pCounter*4];
        yAvg+=seqMBR[pCounter*4+3]-seqMBR[pCounter*4+1];
//if(pCounter==10){
  //printf("\n%d:\t  x0=%d ,  y0=%d ,  x1=%d ,  y1=%d\n", pCounter , seqMBR[4*pCounter], seqMBR[4*pCounter+1], seqMBR[4*pCounter+2], seqMBR[4*pCounter+3]);
  //exit(0);
// }
        pCounter++;
    }
    printf("\nAverage X: %f\n", 1.0*xAvg/pCounter);
    printf("\nAverage Y: %f\n", 1.0*yAvg/pCounter);
    fclose(fid);
    return(pCounter);
}
//==============================================================================

