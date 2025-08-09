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
#include <memory>
#include <vector>
#include <string>

using namespace std;

namespace rtlp {

class Image{
private:
 int W{0}, H{0}; //width->num. columns, height->num. rows
 std::unique_ptr<int[]> data;
 int* ret;  // CUDA memory - keep as raw pointer
 int* cort; // CUDA memory - keep as raw pointer


public:
 Image();
 ~Image();
 // Input Data Array
 inline void SetW(int width) noexcept { W = width; }
 inline void SetH(int height) noexcept { H = height; }
 inline void Set(int c, int r, int val) noexcept { data[r*W+c] = val; }
 inline int Get(int c, int r) const noexcept { return data[r*W+c]; }	
 inline int GetW() const noexcept { return W; }
 inline int GetH() const noexcept { return H; }
 inline int* GetDataPnt() noexcept { return data.get(); }
 inline const int* GetDataPnt() const noexcept { return data.get(); }
 inline int* GetGpuRPnt() noexcept { return ret; }
 inline int* GetGpuCPnt() noexcept { return cort; }
 void SetData(int Wi,int He);
 void SetData(int Wi,int He, int* pnt);
 void SetDataGpuR(int *d);
 void SetDataGpuC(int R, int S);
 void WriteData(const std::string& nf);
 void ReadData(const std::string& nf);
};

} // namespace rtlp
