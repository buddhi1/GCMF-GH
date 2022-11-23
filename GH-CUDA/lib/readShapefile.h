// How to compile 

// g++ readShapefile.cpp -o a.out

// **************************************************

// #include <iostream>
// #include <cstdlib>
// #include <cmath>
// #include <string>
// #include <vector>
// #include <set>

// using namespace std;

// #include "point2D.h"
// #include "polygon.h"

vector<polygon> pPolygons, qPolygons;                 // two input polygons
vector<vector<double>> pPolygonMBRs, qPolygonMBRs;                 // two input polygons

#include <iomanip>
#include<bits/stdc++.h>
#include "Rtree-Topdown-version/librtree/include/index.h"

void printPolygonVector(vector<polygon>& PP) {
  int sum = 0;
  cout << PP.size() << " polygon";
  if (PP.size() > 1) 
    cout << "s";
  cout << " with \n";
  for (int i=0; i<PP.size(); i++) {
    int count = 0;
	cout << "Polygon " << i+1 << "\n";
    for (vertex* V : PP[i].vertices(ALL)){
      count++;
	  cout << setprecision (15) << V->p << endl;
	}
    cout << "my count " << count;
    if (i<PP.size()-1) 
      cout << " + \n";
    sum += count;
  }
  if (PP.size() > 1)
    cout << " = " << sum;
  cout << " vertices\n\n";
}

long double stringToDouble(string s){
	long double num=0.0;
	int precLoc=s.find(".");
	int beforePrec=s.length()-precLoc-1;
	// cout << s.length()-precLoc-1 << endl;

	// after precision
	for(int i=s.length()-1, j=0; i>precLoc; --i, ++j){
		num+=((int)s[i]-48)/pow(10, beforePrec-j);
		cout << s[i] << " - " << num << " " << ((int)s[i]-48)/ pow(10, beforePrec-j) << " ** "<< -pow(10, s.length()-1-j) << endl;
	}
	// before precision
	for(int i=precLoc-1, j=0; i>=0; --i, j++) {
		num+=((int)s[i]-48)*pow(10, j);
	}
	cout << "out " << setprecision (15) << num << endl;
	return num;
}

void loadPolygonFromShapeFile(vector<polygon>& PP, string s, int endOfFile) {
	string line;
	// cout << s << endl;
	ifstream from(s);
	// cout << "line " << line << endl;

	bool polygonReady=false;
	bool polygonStart=false;
	bool vertexFound=false;
	string polygonString="";
	string vertex="";
	string vertex2="";

	point2D v;
    polygon P;
	int count=0;

	do{
		from >> line;
		// cout << line << endl;

		// check if there is comma to find vertices in the polygon
		if (polygonStart && line.find(",")!= std::string::npos) {
			vertex2=line.substr(0, line.find(","));
			vertexFound=true;
		}
		// adding end of the polygon 
		if (polygonStart && line.find("))")!= std::string::npos) {
			vertex2= line.substr(0, line.find("))"));
			polygonReady=true;
			polygonStart=false;

			PP.push_back(P);

			P = polygon(); 
			vertexFound=false;
		}
		
		if(polygonStart && !polygonReady && !vertexFound){
			vertex=line+" ";
		}
		// polygon start
		if (line.find("((")!= std::string::npos){
			vertex= line.substr(line.find("((")+2)+" ";

			polygonStart=true;
			polygonReady=false;
		} 
		
		if(vertexFound){ 
			v=point2D(atof(vertex.c_str()), atof(vertex2.c_str()));
			P.newVertex(v, true);

			vertexFound=false;
		}
		// cout << PP.size() < endOfFile << endl;
	}while((PP.size() < endOfFile || endOfFile == -1) && (!from.eof() || endOfFile != -1));

	// cout << "Num polygons " << PP.size() << endl;
	// printPolygonVector(PP);
}

// for ocean dataset
void loadPolygonFromShapeFile2(vector<polygon>& PP, string s, int endOfFile) {
	string line;
	// cout << s << endl;
	ifstream from(s);
	// cout << "line " << line << endl;

	bool polygonReady=false;
	bool polygonStart=false;
	bool vertexFound=false;
	string polygonString="";
	string vertex="";
	string vertex2="", vertexNew="";

	point2D v;
    polygon P;
	int count=0;

	do{
		from >> line;
		// cout << line << endl;

		// check if there is comma to find vertices in the polygon
		if (polygonStart && line.find("),")== std::string::npos) {
			if (polygonStart && line.find(",")!= std::string::npos) {			
				vertex2=line.substr(0, line.find(","));
				vertex=vertexNew;
				vertexNew=line.substr(line.find(",")+1)+" ";
				vertexFound=true;
				// cout << line << " " << vertex2 << endl;
				// break;
			}
		}
		// adding end of the polygon 
		if (polygonStart && line.find(")))")!= std::string::npos) {
			vertex2= line.substr(0, line.find(")))"));
			polygonReady=true;
			polygonStart=false;

			PP.push_back(P);

			P = polygon(); 
			vertexFound=false;
		} else if (polygonStart && line.find(")")!= std::string::npos) {
			vertex2= line.substr(0, line.find(")"));
			polygonReady=true;
			polygonStart=false;

			PP.push_back(P);

			P = polygon(); 
			vertexFound=false;
		}
		
		// if(polygonStart && !polygonReady && !vertexFound){
		// 	vertex=line+" ";
		// }
		// polygon start
		if (line.find("(((")!= std::string::npos){
			vertexNew= line.substr(line.find("(((")+3)+" ";
			polygonStart=true;
			polygonReady=false;
			// cout << line << " " << vertex << endl;
			// break;
		} else if (line.find("(")!= std::string::npos){
			vertexNew= line.substr(line.find("(")+1)+" ";
			polygonStart=true;
			polygonReady=false;
			// cout << line << " " << vertex << endl;
			// break;
		}
		
		if(vertexFound){ 
			v=point2D(atof(vertex.c_str()), atof(vertex2.c_str()));
			P.newVertex(v, true);

			vertexFound=false;
		}
		// cout << PP.size() < endOfFile << endl;
	}while((PP.size() < endOfFile || endOfFile == -1) && (!from.eof() || endOfFile != -1));

	// cout << "Num polygons " << PP.size() << endl;
	// printPolygonVector(PP);
}

// For Continents
void loadPolygonFromShapeFile3(vector<polygon>& PP, string s, int endOfFile) {
	string line;
	// cout << s << endl;
	ifstream from(s);
	// cout << "line " << line << endl;

	bool polygonReady=false;
	bool polygonStart=false;
	bool vertexFound=false;
	string polygonString="";
	string vertex="", vertexNew="";
	string vertex2="";

	point2D v;
    polygon P;
	int count=0;

	do{
		from >> line;
		// cout << line << endl;

		// check if there is comma to find vertices in the polygon
		if (polygonStart && line.find(")),")== std::string::npos) {
			if (polygonStart && line.find(",")!= std::string::npos) {			
				vertex2=line.substr(0, line.find(","));
				vertex=vertexNew;
				vertexNew=line.substr(line.find(",")+1)+" ";
				vertexFound=true;
				// cout <<"new " << vertexNew << endl;
				// break;
			}
		}
		// adding end of the polygon 
		if (polygonStart && line.find(")))")!= std::string::npos) {
			vertex2= line.substr(0, line.find(")))"));
			polygonReady=true;
			polygonStart=false;
			// cout << "Size polygon " << P.size << endl;
			PP.push_back(P);

			P = polygon(); 
			vertexFound=false;
		} else if (polygonStart && line.find("))")!= std::string::npos) {
			vertex2= line.substr(0, line.find("))"));
			polygonReady=true;
			polygonStart=false;

			// cout << "Size* polygon " << P.size << endl;
			PP.push_back(P);
			// cout<<"*****************************************************\n\n";

			P = polygon(); 
			vertexFound=false;
			// if(PP.size()>3) break;
		}
		
		// if(polygonStart && !polygonReady && !vertexFound){
		// 	vertex=line+" ";
		// }
		// polygon start
		if (line.find("(((")!= std::string::npos){
			vertexNew=line.substr(line.find("(((")+3)+" ";
			polygonStart=true;
			polygonReady=false;
			// cout << line << " " << vertex << endl;
			// break;
		} else if (line.find("((")!= std::string::npos){
			vertexNew=line.substr(line.find("((")+2)+" ";
			polygonStart=true;
			polygonReady=false;
			// cout << line << " " << vertex << endl;
			// break;
		}
		
		if(vertexFound){ 
			v=point2D(atof(vertex.c_str()), atof(vertex2.c_str()));
			// cout<<"test v "<<v.x<<" "<<v.y<<"\n"<<endl;
			P.newVertex(v, true);

			vertexFound=false;
		}
		// cout << PP.size() < endOfFile << endl;
	}while((PP.size() < endOfFile || endOfFile == -1) && (!from.eof() || endOfFile != -1));

	// cout << "Num polygons " << PP.size() << endl;
	// printPolygonVector(PP);
}

vector<double> getMBR(polygon pol){
	vector<double> mbr;
	double maxX, minX, maxY, minY;
	maxX=pol.root->p.x;
	maxY=pol.root->p.y;
	minX=pol.root->p.x;
	minY=pol.root->p.y;

	for (vertex* V : pol.vertices(ALL)){
		if(maxX < V->p.x){
			maxX=V->p.x;
		}
		if(minX > V->p.x){
			minX=V->p.x;
		}
		if(maxY < V->p.y){
			maxY=V->p.y;
		}
		if(minY > V->p.y){
			minY=V->p.y;
		}
	//   cout << setprecision (15) << V->p << endl;
	}
	mbr.push_back(minX);
	mbr.push_back(minY);
	mbr.push_back(maxX);
	mbr.push_back(maxY);
	// uncomment to print in file format
	// printf("%f %f %f %f\n", minX, minY, maxX, maxY);
	return mbr;
}

void loadPolygonMBRs(vector<vector<double>>& pPolygonMBRs, string s, int endOfFile) {
	string line;
	// cout << s << endl;
	ifstream from(s);
	// cout << "line " << line << endl;

	bool polygonReady=false;
	bool polygonStart=false;
	bool vertexFound=false;
	string polygonString="";
	string vertex="";
	string vertex2="";
	vector<double> mbr;

	point2D v;
    polygon P;
	int count=0;

	do{
		from >> line;

		// check if there is comma to find vertices in the polygon
		if (polygonStart && line.find(",")!= std::string::npos) {
			vertex2=line.substr(0, line.find(","));
			vertexFound=true;
		}
		// adding end of the polygon 
		if (polygonStart && line.find("))")!= std::string::npos) {
			vertex2= line.substr(0, line.find("))"));
			polygonReady=true;
			polygonStart=false;
			pPolygonMBRs.push_back(getMBR(P));
			
			P = polygon(); 
			vertexFound=false;
		}
		
		if(polygonStart && !polygonReady && !vertexFound){
			vertex=line+" ";
		}
		// polygon start
		if (line.find("((")!= std::string::npos){
			vertex= line.substr(line.find("((")+2)+" ";

			polygonStart=true;
			polygonReady=false;
		} 
		
		if(vertexFound){ 
			v=point2D(atof(vertex.c_str()), atof(vertex2.c_str()));
			P.newVertex(v, true);

			vertexFound=false;
		}
	}while((pPolygonMBRs.size() < endOfFile || endOfFile == -1) && (!from.eof() || endOfFile != -1));

	// cout << "Num polygons " << pPolygonMBRs.size() << endl;
}

void loadPolygonMBRsFromVector(vector<vector<double>>& pPolygonMBRs, vector<polygon>& PP, int endOfFile) {
	for(int i=0; i<endOfFile; ++i){
		pPolygonMBRs.push_back(getMBR(PP[i]));
		printf("%.17g %.17g  %.17g  %.17g \n", pPolygonMBRs[i][0], pPolygonMBRs[i][1], pPolygonMBRs[i][2], pPolygonMBRs[i][3]);
		// break;
	}
}

int MySearchCallback(int id, void* arg){
	// Note: -1 to make up for the +1 when data was inserted
	printf("Hit data rect %d\n", id-1);
	return 1; // keep going
}

// int main(){
// 	// Linux path
// 	loadPolygonFromShapeFile2(pPolygons, string("../../datasets/ne_10m_ocean.csv"), -1);
// 	loadPolygonFromShapeFile3(qPolygons, string("../../datasets/continents.csv"), -1);
// 	// cout <<pPolygons.size()<<" polygons found"<<endl;
// 	// cout <<qPolygons.size()<<" polygons found"<<endl;
// 	// WSL / Ubuntu path
// 	// loadPolygonFromShapeFile(pPolygons, string("/mnt/d//Datasets/sports/sports"), -1);
// 	// loadPolygonFromShapeFile(PP, string("/mnt/d//Datasets/lakes/lakes"), 1);

// 	// load MBRs from the given polygon data
// 	// loadPolygonMBRs(pPolygonMBRs, string("../../datasets/ne_10m_ocean.csv"), -1);
// 	// loadPolygonMBRs(pPolygonMBRs, string("../../datasets/continents.csv"), -1);

// 	// load MBRs from vectors
// 	// loadPolygonMBRsFromVector(pPolygonMBRs, pPolygons, pPolygons.size());
// 	// loadPolygonMBRsFromVector(qPolygonMBRs, qPolygons, qPolygons.size());

// 	cout<<"P Polygons" << endl;
// 	// order polygon files by the number of vertices
// 	vector<pair <int, int>> vect;
// 	for (int i=0; i<pPolygons.size(); i++){
// 		vect.push_back(make_pair(pPolygons[i].size, i));
// 	}
// 	sort(vect.rbegin(), vect.rend());
// 	// Printing the vector
//     // for (int i=0; i<vect.size(); i++){
//     for (int i=0; i<20; i++){
//         cout << vect[i].first << " " << vect[i].second << endl;
//     }

// 	cout<<"Q Polygons" << endl;
// 	// order polygon files by the number of vertices
// 	vector<pair <int, int>> vectQ;
// 	for (int i=0; i<qPolygons.size(); i++){
// 		vectQ.push_back(make_pair(qPolygons[i].size, i));
// 	}
// 	sort(vectQ.rbegin(), vectQ.rend());
// 	// Printing the vector
//     // for (int i=0; i<vectQ.size(); i++){
//     for (int i=0; i<20; i++){
//         cout << vectQ[i].first << " " << vectQ[i].second << endl;
//     }

// 	// find polygon with most  number of vertices
// 	// int maxPolygonLoc = 0;
// 	// for(int i=1; i<pPolygons.size(); ++i){
// 	// 	if(pPolygons[i].size > pPolygons[maxPolygonLoc].size){
// 	// 		maxPolygonLoc=i;
// 	// 	}
// 	// }
// 	// cout << "Largest polygon has " << pPolygons[maxPolygonLoc].size << " Vertices at " << maxPolygonLoc << endl;

// 	// **use setprecision (15) to show more digits in the given double value in the output on screen
// 	// cout << ">> " << setprecision (15) << stod("2148.232142373") << endl;
// 	return 0;
// }