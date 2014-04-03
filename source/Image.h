#ifndef IMAGE_H
#define IMAGE_H

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

class Image{
private:
 int W, H; //width->num. colonne, height->num. righe
 int* data;
 int* ret;
 int* cort;

public:
 Image();
 ~Image();
 // Array Dati Input
 inline void SetW(int width){W=width;}
 inline void SetH(int height){H=height;}
 inline void Set(int c, int r, int val){data[r*W+c]=val;}
 inline int Get(int c, int r){return data[r*W+c];}	
 inline int GetW(){return W;}
 inline int GetH(){return H;}
 inline int *GetDataPnt(){return data;}
 void SetData(int Wi,int He);
 void SetData(int Wi,int He, int* pnt);
 void WriteData(string nf);
 void ReadData(string nf);

//CUDA
 int *GetGpuRPnt();
 int *GetGpuCPnt();
 void SetDataGpuR(int *d);
 void SetDataGpuC(int R, int S);

};

#endif
