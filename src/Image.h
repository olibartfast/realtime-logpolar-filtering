#pragma once

#if  __unix__
 #define CLRSCR "clear"
 #define Ttime (1000.0)	

#else
 #define CLRSCR "cls"
 #define Ttime 1.0	

#endif

#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <ctime>
#include <cassert>
#include <cmath>

using namespace std;

namespace rtlp {

class Image{
private:
 int W, H; //width->num. columns, height->num. rows
 int* data;
 int* ret;
 int* cort;


public:
 Image();
 ~Image();
 // Input Data Array
 inline void SetW(int width){W=width;}
 inline void SetH(int height){H=height;}
 inline void Set(int c, int r, int val){data[r*W+c]=val;}
 inline int Get(int c, int r){return data[r*W+c];}	
 inline int GetW(){return W;}
 inline int GetH(){return H;}
 inline int *GetDataPnt(){return data;}
 inline int* GetGpuRPnt(){return ret;}
 inline int* GetGpuCPnt(){return cort;}
 void SetData(int Wi,int He);
 void SetData(int Wi,int He, int* pnt);
 void SetDataGpuR(int *d);
 void SetDataGpuC(int R, int S);
 void WriteData(string nf);
 void ReadData(string nf);
};

} // namespace rtlp
