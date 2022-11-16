#include <iostream>
#include<bits/stdc++.h>
#include <vector>
#include <chrono>

#include "lib/polyclip.cpp"
#include "ghcuda.h"

using namespace std::chrono;

#include "lib/readShapefile.cpp"


int argn;
string outputFile=string("results/Fig-def-R.poly");

// read from shape files
void readInputFromShapeFiles(double **baseCoords, double **overlayCoords, string inputShp1, int PPID, string inputShp2, int QQID){
  int sizeP=PPTmp[PPID].size, sizeQ=QQTmp[QQID].size;
  if(sizeP<sizeQ){
    PP.push_back(QQTmp[QQID]);
    QQ.push_back(PPTmp[PPID]);
    sizeP=QQTmp[QQID].size;
    sizeQ=PPTmp[PPID].size;

    if(DEBUG_INFO_PRINT){
      cout<<"Shape file1: "<<inputShp1<<" PPID: "<<PPID<<endl;;
      cout<<"Shape file2: "<<inputShp2<<" QQID: "<<QQID<<endl;
      cout << "PP and QQ swapped since QQ> PP\nNew PP Polygon size " << sizeP;
      cout << " New QQ Polygon size " << sizeQ << endl;
    }
  } else {
    PP.push_back(PPTmp[PPID]);
    QQ.push_back(QQTmp[QQID]);

    if(DEBUG_INFO_PRINT){
      cout<<"Shape file1: "<<inputShp1<<" PPID: "<<PPID<<endl;;
      cout<<"Shape file2: "<<inputShp2<<" QQID: "<<QQID<<endl;
      cout << "PP Polygon size " << sizeP;
      cout << " QQ Polygon size " << sizeQ << endl;
    }
  }

  *baseCoords=(double *)malloc(2*sizeP*sizeof(double));
  *overlayCoords=(double *)malloc(2*sizeQ*sizeof(double));

  int i=0;
  // copy polygon P values
  for (vertex* V : PP[0].vertices(ALL)){
    *(*baseCoords+i++) = V->p.x;
    *(*baseCoords+i++) = V->p.y;
	}
  if(DEBUG_INFO_PRINT) cout<<"PP Count "<<i;

  i=0;
  // copy polygon Q values
  for (vertex* V : QQ[0].vertices(ALL)){
    *(*overlayCoords+i++) = V->p.x;
    *(*overlayCoords+i++) = V->p.y;
	}
  if(DEBUG_INFO_PRINT) cout<<" QQ Count "<<i<<endl;
}

// get CMBR for PP and QQ
void getCMBR(double *cmbr){
  vector<double> PPMBR;
  vector<double> QQMBR;
  // double *cmbr; //minx, miny, maxx, maxy
  double minX, minY, maxX, maxY;

  PPMBR=getMBR(PP[0]);
  QQMBR=getMBR(QQ[0]);

  if(DEBUG_INFO_PRINT){
    cout<<"MBR_P ["<<PPMBR[0]<<", "<<PPMBR[1]<<", "<<PPMBR[2]<<", "<<PPMBR[3]<<endl;
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
void readPolygons(int argc, char* argv[], double **baseCoords, double **overlayCoords, string inputShp1, int PPID, string inputShp2, int QQID){
  // check input parameters
  if (argc < 4) {
    readInputFromShapeFiles(baseCoords, overlayCoords, inputShp1, PPID, inputShp2, QQID);
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
    gpc_read_polygon(pfile, baseCoords, &sizeP, "PP");
    gpc_read_polygon(qfile, overlayCoords, &sizeQ, "QQ");
    // */

    // -------------------------------------------------------------------------------------------
    // Alternate 2 -> PHASE:1 read input polygons from given XY format with , and ; seperators
    // -------------------------------------------------------------------------------------------
    /*
    cout << "\nP "; loadPolygon(PP,string(argv[argn++]));
    cout <<   "Q "; loadPolygon(QQ,string(argv[argn++]));
    int i=0;
    // copy polygon P values
    for (vertex* V : PP[0].vertices(ALL)){
      *(*polyPX+i) = V->p.x;
      *(*polyPY+i++) = V->p.y;
      // cout << "--- " << setprecision (15) << polyPX[i-1] << ", " << polyPY[i-1] << endl;
    }
    if(DEBUG_INFO_PRINT) cout<<"PP Count "<<i;

    i=0;
    // copy polygon Q values
    for (vertex* V : QQ[0].vertices(ALL)){
      *(*polyQX+i) = V->p.x;
      *(*polyQY+i++) = V->p.y;
      // cout << "--- " << setprecision (15) << V->p.x << endl;
    }
    if(DEBUG_INFO_PRINT) cout<<" QQ Count "<<i<<endl;
    // -------------------------------------------------------------------------------------------
  */
  outputFile=string(argv[argn]);
  }
}

// handles polygons without holes
void regularPolygonHandler(double *baseCoord, double *overlayCoords){
  // -------------------------------------------------------------------------------------------
  // PHASE:2 calculate intersections using GPU acceleration
  // -------------------------------------------------------------------------------------------
  double *intersectionsP, *intersectionsQ;
  int countNonDegenIntP, countNonDegenIntQ, *initLabelsP, *initLabelsQ, *neighborP, *neighborQ;
  int *alphaValuesP, *alphaValuesQ;
  vertex *tmpVertex, *current;
  double *cmbr;
  cmbr=(double *)malloc(4*sizeof(double));
  getCMBR(cmbr);

  calculateIntersections(
      baseCoord, 
      overlayCoords, 
      PP[0].size, QQ[0].size, cmbr, 
      &countNonDegenIntP, &countNonDegenIntQ, 
      &intersectionsP, &intersectionsQ, &alphaValuesP, &alphaValuesQ,
      &initLabelsP, &initLabelsQ, 
      &neighborP, &neighborQ);
  // -------------------------------------------------------------------------------------------
// return 0;
  // -------------------------------------------------------------------------------------------
  // Polygon P: (PP)insert intersection vertices and change alpha value in the degenerate cases
  // -------------------------------------------------------------------------------------------
  int i=0, j=0, pi=0;
  int intersectionPArrayMax=countNonDegenIntP*2;
  PPVertexPointers=new vertex*[countNonDegenIntP];

  vertex* V=PP[0].root;
  for(int ii=0; ii<=PP[0].size; ii++){
    current=V;
    while(*(intersectionsP+(i%intersectionPArrayMax))!=V->p.x || *(intersectionsP+((i+1)%intersectionPArrayMax))!=V->p.y){
      tmpVertex=new vertex(*(intersectionsP+i), *(intersectionsP+i+1));
      tmpVertex->label=(IntersectionLabel)(*(initLabelsP+(i/2)));
      tmpVertex->source=false;
      tmpVertex->intersection=true;
      tmpVertex->next=current;
      current->prev->next=tmpVertex;
      tmpVertex->prev=current->prev;
      current->prev=tmpVertex;
      PPVertexPointers[pi++]=tmpVertex;
      i+=2;
    }
    if(ii<PP[0].size){ 
      PPVertexPointers[pi]=V;
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
  // Polygon Q: (QQ)insert intersection vertices and change alpha value in the degenerate cases
  // -------------------------------------------------------------------------------------------
  i=0;
  V=QQ[0].root;
  int intersectionQArrayMax=countNonDegenIntQ*2;
  for(int ii=0; ii<=QQ[0].size; ++ii){
    current=V;
    while(*(intersectionsQ+(i%intersectionQArrayMax))!=V->p.x || *(intersectionsQ+((i+1)%intersectionQArrayMax))!=V->p.y){
      tmpVertex=new vertex(*(intersectionsQ+i), *(intersectionsQ+i+1));
      tmpVertex->label=(IntersectionLabel)(*(initLabelsQ+(i/2)));
      tmpVertex->source=false;
      tmpVertex->intersection=true;
      tmpVertex->next=current;
      current->prev->next=tmpVertex;
      tmpVertex->prev=current->prev;
      current->prev=tmpVertex;
      tmpVertex->neighbour=PPVertexPointers[(*(neighborQ+(i/2)))-1];
      PPVertexPointers[(*(neighborQ+(i/2)))-1]->neighbour=tmpVertex;
      i+=2;
    }
    if(ii<QQ[0].size){
      V->label=(IntersectionLabel)(*(initLabelsQ+(i/2)));
      if(*(alphaValuesQ+(i/2))!=-100){ 
        V->intersection=true;
        V->neighbour=PPVertexPointers[(*(neighborQ+(i/2)))-1];
        PPVertexPointers[(*(neighborQ+(i/2)))-1]->neighbour=V;
      }
    }
    i+=2;
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
}

void GH_CUDA(double *baseCoords, double *overlayCoords){
  high_resolution_clock::time_point start, end, start1, end1, start2, end2, start3, end3;

  if(DEBUG_TIMING) start = high_resolution_clock::now();

  if(DEBUG_TIMING) start3 = high_resolution_clock::now();
  regularPolygonHandler(baseCoords, overlayCoords);
  if(DEBUG_TIMING) end3 = high_resolution_clock::now();// -------------------------------------------------------------------------------------------

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
    savePolygon(RR, outputFile);
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
}

int ghcuda(int PPID_list[], int QQID_list[]){
// int main(int argc, char* argv[]){

  double *baseCoords, *overlayCoords;
  // [0, 36, 2742, 2741, 5978, | 2854, 2737]
  // int PPID_list[]={0, 36}; //ne_10m_ocean
  int PPID=36;  //defines the end of file. Use -1 to read the complete file
  string inputShp1=string("../datasets/ne_10m_ocean.csv");
  loadPolygonFromShapeFile2(PPTmp, inputShp1, PPID+1);

  // [521, 1048, 1202, 1661, 1886, | 1524, 54, 1081, 1193]
  // string inputShp2=string("../datasets/continents.csv");
  // int QQID=521; //continents
  // loadPolygonFromShapeFile2(QQTmp, inputShp2, QQID+1);

  // time these for paper
  // ocean, land [2742, 30] [2742, 42]
  // [4, 1, 0, 33, 30, 3, | 42, 25, 8, 19]
  string inputShp2=string("../datasets/ne_10m_land.csv");
  // int QQID_list[]={4, 1}; //ne_10m_land
  int QQID=4;  //defines the end of file. Use -1 to read the complete file
  loadPolygonFromShapeFile2(QQTmp, inputShp2, QQID+1);
  // readPolygons(argc, argv, &baseCoords, &overlayCoords, inputShp1, PPID, inputShp2, QQID);

  for(int cid=0; cid<2; ++cid){
    readInputFromShapeFiles(&baseCoords, &overlayCoords, inputShp1, PPID_list[cid], inputShp2, QQID_list[cid]);
    GH_CUDA(baseCoords, overlayCoords);
    free(baseCoords);
    free(overlayCoords);
    PP.clear();
    QQ.clear();
  }
  
  return 0;
}