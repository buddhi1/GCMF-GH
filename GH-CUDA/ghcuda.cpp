// #include <iostream>
#include<bits/stdc++.h>
// #include <vector>
#include <chrono>

#include "lib/polyclip.h"
#include "ghcuda.h"
#include "lib/readShapefile.h"

using namespace std::chrono;


int argn;
string outputFile=string("results/Results-def-R.poly");

// read from shape files
void readInputFromShapeFiles(coord_t **baseCoords, coord_t **overlayCoords, string inputShp1, int pID, string inputShp2, int qID){
  int sizeP=pPolygons[pID].size, sizeQ=qPolygons[qID].size;
  if(sizeP<sizeQ){
    pPolygon.push_back(qPolygons[qID]);
    qPolygon.push_back(pPolygons[pID]);
    sizeP=qPolygons[qID].size;
    sizeQ=pPolygons[pID].size;

    if(DEBUG_INFO_PRINT){
      cout<<"Shape file1: "<<inputShp1<<" pID: "<<pID<<endl;;
      cout<<"Shape file2: "<<inputShp2<<" qID: "<<qID<<endl;
      cout << "pPolygon and qPolygon swapped since qPolygon> pPolygon\nNew pPolygon Polygon size " << sizeP;
      cout << " New qPolygon Polygon size " << sizeQ << endl;
    }
  } else {
    pPolygon.push_back(pPolygons[pID]);
    qPolygon.push_back(qPolygons[qID]);

    if(DEBUG_INFO_PRINT){
      cout<<"Shape file1: "<<inputShp1<<" pID: "<<pID<<endl;;
      cout<<"Shape file2: "<<inputShp2<<" qID: "<<qID<<endl;
      cout << "pPolygon Polygon size " << sizeP;
      cout << " qPolygon Polygon size " << sizeQ << endl;
    }
  }

  *baseCoords=(coord_t *)malloc(2*sizeP*sizeof(coord_t));
  *overlayCoords=(coord_t *)malloc(2*sizeQ*sizeof(coord_t));

  int i=0;
  // copy polygon P values
  for (vertex* V : pPolygon[0].vertices(ALL)){
    *(*baseCoords+i++) = V->p.x;
    *(*baseCoords+i++) = V->p.y;
	}
  if(DEBUG_INFO_PRINT) cout<<"pPolygon Count "<<i;

  i=0;
  // copy polygon Q values
  for (vertex* V : qPolygon[0].vertices(ALL)){
    *(*overlayCoords+i++) = V->p.x;
    *(*overlayCoords+i++) = V->p.y;
	}
  if(DEBUG_INFO_PRINT) cout<<" qPolygon Count "<<i<<endl;
}

// load polygon data from polygons vector to the single polygon linked list
int loadPolygonDataFromLayer(int pID, int qID){
  int sizeP=pPolygons[pID].size, sizeQ=qPolygons[qID].size;
  if(sizeP<sizeQ){
    pPolygon.push_back(qPolygons[qID]);
    qPolygon.push_back(pPolygons[pID]);
    sizeP=qPolygons[qID].size;
    sizeQ=pPolygons[pID].size;

    if(DEBUG_INFO_PRINT){
      cout<<"\nGCMF input data share: pID: "<<pID<<endl;;
      cout<<"GCMF input data share: qID: "<<qID<<endl;
      cout << "pPolygon and qPolygon swapped since qPolygon> pPolygon\nNew pPolygon Polygon size " << sizeP;
      cout << " New qPolygon Polygon size " << sizeQ << endl;
    }
    return 1;
  } else {
    pPolygon.push_back(pPolygons[pID]);
    qPolygon.push_back(qPolygons[qID]);

    if(DEBUG_INFO_PRINT){
      cout<<"\n*GCMF input data share: pID: "<<pID<<endl;;
      cout<<"GCMF input data share: qID: "<<qID<<endl;
      cout << "pPolygon Polygon size " << sizeP;
      cout << " qPolygon Polygon size " << sizeQ << endl;
    }
    return 0;
  }
}

// get CMBR for pPolygon and qPolygon
void getCMBR(coord_t *cmbr){
  vector<coord_t> PPMBR;
  vector<coord_t> QQMBR;
  // double *cmbr; //minx, miny, maxx, maxy
  double minX, minY, maxX, maxY;

  PPMBR=getMBR(pPolygon[0]);
  QQMBR=getMBR(qPolygon[0]);

  if(DEBUG_INFO_PRINT){
    cout<<setprecision(13)<<"MBR_P ["<<PPMBR[0]<<", "<<PPMBR[1]<<", "<<PPMBR[2]<<", "<<PPMBR[3]<<endl;
    cout<<"MBR_Q ["<<QQMBR[0]<<", "<<QQMBR[1]<<", "<<QQMBR[2]<<", "<<QQMBR[3]<<endl;
  }

  // check intersection between MBRs
  if(PPMBR[0]>QQMBR[2] || PPMBR[2]<QQMBR[0]){
    printf("No Overlap between polygons\n");
    exit(0);
  }
  if(PPMBR[1]>QQMBR[3] || PPMBR[3]<QQMBR[1]){
    printf("No Overlap between polygons\n");
    exit(0);
  }

  cmbr[0]=max(PPMBR[0], QQMBR[0]);
  cmbr[1]=max(PPMBR[1], QQMBR[1]);
  cmbr[2]=min(PPMBR[2], QQMBR[2]);
  cmbr[3]=min(PPMBR[3], QQMBR[3]);
  if(DEBUG_INFO_PRINT){
    cout<<"CMBR ["<<cmbr[0]<<", "<<cmbr[1]<<", "<<cmbr[2]<<", "<<cmbr[3]<<endl;
  }
}

// read polygons into vectors
void readPolygons(int argc, char* argv[], coord_t **baseCoords, coord_t **overlayCoords, string inputShp1, int pID, string inputShp2, int qID){
  // check input parameters
  if (argc < 4) {
    readInputFromShapeFiles(baseCoords, overlayCoords, inputShp1, pID, inputShp2, qID);
  }else{
    argn = 1;
    if (string(argv[1]) == "-union") {
      cout << "\n!!! computing UNION instead of INTERSECTION !!!\n";
      UNION = true;
      argn++;
    }
    // ******** MAKE SURE sizeP>sizeQ
    int sizeP=0, sizeQ=0, i=0;

    // -------------------------------------------------------------------------------------------
    // Alternate 1 -> PHASE:1 read input polygons from polygon XY format file
    // -------------------------------------------------------------------------------------------
    // /*
    FILE *pfile, *qfile;
    pfile=fopen(argv[argn++], "r");
    qfile=fopen(argv[argn++], "r");
    gpc_read_polygon(pfile, baseCoords, &sizeP, "pPolygon");
    gpc_read_polygon(qfile, overlayCoords, &sizeQ, "qPolygon");
    // */

    // -------------------------------------------------------------------------------------------
    // Alternate 2 -> PHASE:1 read input polygons from given XY format with , and ; seperators
    // -------------------------------------------------------------------------------------------
    /*
    cout << "\nP "; loadPolygon(pPolygon,string(argv[argn++]));
    cout <<   "Q "; loadPolygon(qPolygon,string(argv[argn++]));
    int i=0;
    // copy polygon P values
    for (vertex* V : pPolygon[0].vertices(ALL)){
      *(*polyPX+i) = V->p.x;
      *(*polyPY+i++) = V->p.y;
      // cout << "--- " << setprecision (15) << polyPX[i-1] << ", " << polyPY[i-1] << endl;
    }
    if(DEBUG_INFO_PRINT) cout<<"pPolygon Count "<<i;

    i=0;
    // copy polygon Q values
    for (vertex* V : qPolygon[0].vertices(ALL)){
      *(*polyQX+i) = V->p.x;
      *(*polyQY+i++) = V->p.y;
      // cout << "--- " << setprecision (15) << V->p.x << endl;
    }
    if(DEBUG_INFO_PRINT) cout<<" qPolygon Count "<<i<<endl;
    // -------------------------------------------------------------------------------------------
  */
  outputFile=string(argv[argn]);
  }
}

// handles polygons without holes
int regularPolygonHandler(coord_t *bCoord, coord_t *oCoords, int *pBVNum, long *pBVPSNum, int *pOVNum, long *pOVPSNum, int bPID, int oPID, int swapped){
// void regularPolygonHandler(coord_t *baseCoord, coord_t *overlayCoords){
  // -------------------------------------------------------------------------------------------
  // PHASE:2 calculate intersections using GPU acceleration
  // -------------------------------------------------------------------------------------------
  coord_t *intersectionsP, *intersectionsQ;
  int countNonDegenIntP, countNonDegenIntQ, *initLabelsP, *initLabelsQ, *neighborP, *neighborQ;
  int *alphaValuesP, *alphaValuesQ, invalid=0;
  vertex *tmpVertex, *current;
  coord_t *cmbr;
  cmbr=(coord_t *)malloc(4*sizeof(coord_t));
  getCMBR(cmbr);

  if(swapped){
    calculateIntersections(
        oCoords,
        bCoord, 
        pPolygon[0].size, qPolygon[0].size, cmbr, 
        pOVNum, pOVPSNum, pBVNum, pBVPSNum, oPID, bPID,
        &countNonDegenIntQ, &countNonDegenIntP, 
        &intersectionsQ,  &intersectionsP, &alphaValuesQ, &alphaValuesP,
        &initLabelsQ, &initLabelsP, 
        &neighborQ, &neighborP, &invalid);
  }else{
    calculateIntersections(
        bCoord, 
        oCoords, 
        pPolygon[0].size, qPolygon[0].size, cmbr, 
        pBVNum, pBVPSNum, pOVNum, pOVPSNum, bPID, oPID,
        &countNonDegenIntP, &countNonDegenIntQ, 
        &intersectionsP, &intersectionsQ, &alphaValuesP, &alphaValuesQ,
        &initLabelsP, &initLabelsQ, 
        &neighborP, &neighborQ, &invalid);
  }
  // -------------------------------------------------------------------------------------------
  cout<<"invalid "<<invalid<<endl;
  if(invalid) return invalid;

  // printf("\nneighbor array %d\n", countNonDegenIntP);
  // for(int cc=0; cc<countNonDegenIntP; ++cc){
  //   printf("%d-%d ", cc, neighborP[cc]);
  // }
  printf("\nneighbor array %d\n", countNonDegenIntQ);
  for(int cc=0; cc<countNonDegenIntQ; ++cc){
    printf("%d-%d ", cc, neighborQ[cc]);
  }

  // -------------------------------------------------------------------------------------------
  // Polygon P: (pPolygon)insert intersection vertices and change alpha value in the degenerate cases
  // -------------------------------------------------------------------------------------------
  int i=0, j=0, pi=0;
  int intersectionPArrayMax=countNonDegenIntP*2;
  qPolygonVertexPointers=new vertex*[countNonDegenIntP];
  vertex* V=pPolygon[0].root;
  if(DEBUG_INFO_PRINT) printf(" Polygon P copying \n");
  // printf("\n root V (%f, %f)\n", V->p.x, V->p.y); 
  
  for(int ii=0; ii<=pPolygon[0].size; ii++){
    current=V;
    while((fabs(*(intersectionsP+(i%intersectionPArrayMax))-V->p.x) > EPSILON) || (fabs(*(intersectionsP+((i+1)%intersectionPArrayMax))-V->p.y) > EPSILON)){
    // while(*(intersectionsP+(i%intersectionPArrayMax))!=V->p.x || *(intersectionsP+((i+1)%intersectionPArrayMax))!=V->p.y){
      tmpVertex=new vertex(*(intersectionsP+i), *(intersectionsP+i+1));
      tmpVertex->label=(IntersectionLabel)(*(initLabelsP+(i/2)));
      tmpVertex->source=false;
      tmpVertex->intersection=true;
      tmpVertex->next=current;
      current->prev->next=tmpVertex;
      tmpVertex->prev=current->prev;
      current->prev=tmpVertex;
      qPolygonVertexPointers[pi++]=tmpVertex;
      i+=2;
      // cout<<"from p copy pi="<<pi-1<<endl;
    }
    if(ii<pPolygon[0].size){ 
      qPolygonVertexPointers[pi]=V;
      pi++;
      V->label=(IntersectionLabel)(*(initLabelsP+(i/2)));
      if(*(alphaValuesP+(i/2))!=-100){
        V->intersection=true;
      }
    }

    i+=2;
    V=current->next;
  }
  // -------------------------------------------------------------------------------------------
  // Polygon Q: (qPolygon)insert intersection vertices and change alpha value in the degenerate cases
  // -------------------------------------------------------------------------------------------
  i=0;
  V=qPolygon[0].root;
  int intersectionQArrayMax=countNonDegenIntQ*2;
  if(DEBUG_INFO_PRINT) printf(" Polygon Q copying (%d) \n", qPolygon[0].size);
  
  for(int ii=0; ii<=qPolygon[0].size; ++ii){
    current=V;
    // cout<<"P* Copying "<<ii<<endl;
    while((fabs(*(intersectionsQ+(i%intersectionQArrayMax))-V->p.x) > EPSILON) || (fabs(*(intersectionsQ+((i+1)%intersectionQArrayMax))-V->p.y) > EPSILON)){
    // while(*(intersectionsQ+(i%intersectionQArrayMax))!=V->p.x || *(intersectionsQ+((i+1)%intersectionQArrayMax))!=V->p.y){
      // cout<<"q$$$$** Copying "<<i<<endl;
      tmpVertex=new vertex(*(intersectionsQ+i), *(intersectionsQ+i+1));
      tmpVertex->label=(IntersectionLabel)(*(initLabelsQ+(i/2)));
      tmpVertex->source=false;
      tmpVertex->intersection=true;
      tmpVertex->next=current;
      current->prev->next=tmpVertex;
      tmpVertex->prev=current->prev;
      current->prev=tmpVertex;
      // cout<<"1######** Copying "<<(*(neighborQ+(i/2)))-1<<endl;
      // cout<< fabs(*(intersectionsQ+((i+1)%intersectionQArrayMax))-V->p.y) <<"q++++ Copying "<<i<<" "<< fabs(*(intersectionsQ+(i%intersectionQArrayMax))-V->p.x) <<endl;
      tmpVertex->neighbour=qPolygonVertexPointers[(*(neighborQ+(i/2)))-1];
      // cout<< fabs(*(intersectionsQ+((i+1)%intersectionQArrayMax))-V->p.y) <<"q$$$$===** Copying "<<i<<" "<< fabs(*(intersectionsQ+(i%intersectionQArrayMax))-V->p.x) <<endl;
      qPolygonVertexPointers[(*(neighborQ+(i/2)))-1]->neighbour=tmpVertex;
      // cout<<"q$$$$-----** Copying "<<i<<endl;
      i+=2;
      // cout<<"P** Copying "<<i<<endl;
    }
    // cout<<"P*++* Copying "<<i<<endl;
    if(ii<qPolygon[0].size){
      V->label=(IntersectionLabel)(*(initLabelsQ+(i/2)));
      if(*(alphaValuesQ+(i/2))!=-100){ 
        V->intersection=true;
        V->neighbour=qPolygonVertexPointers[(*(neighborQ+(i/2)))-1];
        qPolygonVertexPointers[(*(neighborQ+(i/2)))-1]->neighbour=V;
      }
    }
    i+=2;
    // cout<<"P*-----* Copying "<<i<<endl;
    V=current->next;
  }
  if(DEBUG_INFO_PRINT) printf("Copying completed");
  // -------------------------------------------------------------------------------------------
  free(intersectionsP);
  free(intersectionsQ);
  free(initLabelsP);
  free(initLabelsQ);
  free(neighborP);
  free(neighborQ);
  free(alphaValuesP);
  free(alphaValuesQ);
  // free(tmpVertex);
  // free(current);
  free(cmbr);
  free(qPolygonVertexPointers);

  return invalid;
}

int GH_CUDA(coord_t *bCoords, coord_t *oCoords, int *pBVNum, long *pBVPSNum, int *pOVNum, long *pOVPSNum, int bPID, int oPID, int swapped){
// void GH_CUDA(coord_t *baseCoords, coord_t *overlayCoords){
  high_resolution_clock::time_point start, end, start1, end1, start2, end2, start3, end3;

  if(DEBUG_TIMING) start = high_resolution_clock::now();

  if(DEBUG_TIMING) start3 = high_resolution_clock::now();
  int invalid=regularPolygonHandler(bCoords, oCoords, pBVNum, pBVPSNum, pOVNum, pOVPSNum, bPID, oPID, swapped);
  if(DEBUG_TIMING) end3 = high_resolution_clock::now();// -------------------------------------------------------------------------------------------
  if(invalid) return invalid;
  
  // PHASE: 3
  if(DEBUG_TIMING) start1 = high_resolution_clock::now();
  labelIntersections();
  if(DEBUG_TIMING) end1 = high_resolution_clock::now();// -------------------------------------------------------------------------------------------

  // PHASE: 4
  if(DEBUG_TIMING) start2 = high_resolution_clock::now();
  createResult();
  if(DEBUG_TIMING) end2 = high_resolution_clock::now();// -------------------------------------------------------------------------------------------
  
  if(DEBUG_TIMING) end = high_resolution_clock::now();
  // post-processing
  cleanUpResult();

  // write output polygon
  if(DEBUG_INFO_PRINT) {
    cout << "R ";
    savePolygon(resultPolygon, outputFile);
  }
  if(DEBUG_TIMING){
    auto duration = duration_cast<microseconds>(end - start);
    auto duration1 = duration_cast<microseconds>(end1 - start1);
    auto duration2 = duration_cast<microseconds>(end2 - start2);
    auto duration3 = duration_cast<microseconds>(end3 - start3);
    cout<<"All time in microseconds\nTime: Total : " << fixed<< duration.count() << setprecision(10) << endl;
    cout<<"Total intersection: " << fixed<< duration3.count() << setprecision(10) << endl;
    cout<<"Sequential labeling: " << fixed<< duration1.count() << setprecision(10) << endl;
    cout<<"Sequential labeling: " << fixed<< duration2.count() << setprecision(10) << endl;
  }
  return invalid;
}

int ghcuda(int pIDList[], int qIDList[], int totalNumPairs,
          /*coord_t *baseCoords, coord_t *overlayCoords,  */                 
          coord_t *bCoords, coord_t *oCoords,
          int *pBVNum, long *pBVPSNum, int *pOVNum, long *pOVPSNum){
  pIDList[0]={18};
  qIDList[0]={670};
  int swapped=0;
  for(int cid=0, processedID=1; cid<1; ++cid){
  // for(int cid=0, processedID=1; cid<totalNumPairs; ++cid){
    //skip list for error handling
    // if(
    //   (qIDList[cid]==11771) ||
    //   (qIDList[cid]==11754) ||
    //   (qIDList[cid]==11770) ||
    //   (qIDList[cid]==11770) ||
    //   (qIDList[cid]==11780) ||
    //   (qIDList[cid]==11707) ||
    //   (qIDList[cid]==11704) ||
    //   (qIDList[cid]==11827) ||
    //   (qIDList[cid]==9609) ||

    //   // (pIDList[cid]==177) || 
    //   (pIDList[cid]==175) || 
    //   (pIDList[cid]==174) || 
    //   (pIDList[cid]==172) || 
    //   (pIDList[cid]==171) || 
    //   (pIDList[cid]==170) || 
    //   (pIDList[cid]==166) || 
    //   (pIDList[cid]==158) || 
    //   (pIDList[cid]==145) || 
    //   (pIDList[cid]==133) || 
    //   (pIDList[cid]==132) || 
    //   (pIDList[cid]==131) || 
    //   (pIDList[cid]==130) || 
    //   (pIDList[cid]==120) || 
    //   (pIDList[cid]==118) || 
    //   (pIDList[cid]==115 && qIDList[cid]==5416) || 
    //   pIDList[cid]==112 || 
    //   pIDList[cid]==110 || 
    //   (pIDList[cid]==105) || 
    //   pIDList[cid]==104 || 
    //   pIDList[cid]==103 || 
    //   pIDList[cid]==72 || 
    //   pIDList[cid]==44 || 
    //   pIDList[cid]==27 /* || 
      // pIDList[cid]==18 )
    //   pIDList[cid]==22 || 
    //   pIDList[cid]==41) 
      // {
      //   continue; 
      // }

    // readInputFromShapeFiles(&baseCoords, &overlayCoords, inputShp1, pIDList[cid], inputShp2, qIDList[cid]);
    swapped=loadPolygonDataFromLayer(pIDList[cid], qIDList[cid]);
   
    // printf("\n*** Pair  %d %d***\n", pPolygon[0].size, qPolygon[0].size);
    // if((pPolygon[0].size%2 != qPolygon[0].size%2) /*|| (pPolygon[0].size<100 || qPolygon[0].size<100)*/) {
    //   pPolygon.clear();
    //   qPolygon.clear();
    //   continue;
    // }
    printf("\n\n*** Pair ID: %d, Processing Number: %d***\n", cid, processedID++);
    printf(" [Pair invalid=%d]\n\n", 
            GH_CUDA(bCoords, oCoords, pBVNum, pBVPSNum, pOVNum, pOVPSNum, pIDList[cid], qIDList[cid], swapped));

    // GH_CUDA(bCoords, oCoords, pBVNum, pBVPSNum, pOVNum, pOVPSNum, pIDList[cid], qIDList[cid]);
    // GH_CUDA(baseCoords, overlayCoords);
    // free(baseCoords);
    // free(overlayCoords);
    pPolygon.clear();
    qPolygon.clear();
  }
  
  return 0;
}

// =========================================================================================================
// Old main method. GCMF does not need I/O read anymore from GH code. I/O is reused from the GCMF file read

// int ghcuda(int pIDList[], int qIDList[], int totalNumPairs){
/*int main(int argc, char* argv[]){

  coord_t *baseCoords, *overlayCoords;
  // [0, 36, 2742, 2741, 5978, | 2854, 2737]
  int pIDList[]={0, 36}; //ne_10m_ocean
  int pID=36;  //defines the end of file. Use -1 to read the complete file
  string inputShp1=string("../../datasets/ne_10m_ocean.csv");
  loadPolygonFromShapeFile2(pPolygons, inputShp1, pID+1);

  // [521, 1048, 1202, 1661, 1886, | 1524, 54, 1081, 1193]
  // string inputShp2=string("../datasets/continents.csv");
  // int qID=521; //continents
  // loadPolygonFromShapeFile2(qPolygons, inputShp2, qID+1);

  // time these for paper
  // ocean, land [2742, 30] [2742, 42]
  // [4, 1, 0, 33, 30, 3, | 42, 25, 8, 19]
  string inputShp2=string("../../datasets/ne_10m_land.csv");
  int qIDList[]={4, 1}; //ne_10m_land
  int qID=4;  //defines the end of file. Use -1 to read the complete file
  loadPolygonFromShapeFile2(qPolygons, inputShp2, qID+1);
  // readPolygons(argc, argv, &baseCoords, &overlayCoords, inputShp1, pID, inputShp2, qID);

  for(int cid=0; cid<2; ++cid){
    readInputFromShapeFiles(&baseCoords, &overlayCoords, inputShp1, pIDList[cid], inputShp2, qIDList[cid]);
    GH_CUDA(baseCoords, overlayCoords);
    free(baseCoords);
    free(overlayCoords);
    pPolygon.clear();
    qPolygon.clear();
  }
  
  return 0;
}*/