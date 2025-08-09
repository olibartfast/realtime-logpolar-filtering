#include "rtlp/core/Image.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace rtlp {
namespace core {

Image::Image() : ret(nullptr), cort(nullptr) {}


Image::~Image(){
 // data is automatically cleaned up by unique_ptr
 if(ret != nullptr)
 cudaFree(ret);
 if(cort != nullptr)
 cudaFree(cort);
}


void Image::SetData(int Wi,int He){
 data = std::make_unique<int[]>(Wi*He);
 W = Wi;
 H = He;
}


void Image::SetData(int Wi,int He, int* pnt){
 W = Wi; 
 H = He;
 data = std::make_unique<int[]>(Wi*He);
 for(int row=0; row<He; row++)
  for(int col=0; col<Wi; col++)
   data[row*Wi+col] = pnt[row*Wi+col];
}



void Image::SetDataGpuR(int *d){
 if(ret != nullptr)
 cudaFree(ret);
 cudaMalloc((void**)&ret, W*H*sizeof(int));
 cudaMemcpy(ret, d, W*H*sizeof(int), cudaMemcpyHostToDevice);
}

void Image::SetDataGpuC(int R, int S){
 if(cort != nullptr)
 cudaFree(cort);
 cudaMalloc((void**)&cort, R*S*sizeof(int));
}

void Image::ReadData(const std::string& nf)
{
 auto img = cv::imread(nf);
 if (!img.data){ 
  cout<<"Cannot read the image"<<endl;
  return;
 }	
 cv::Mat imgR;
 	cvtColor(img, imgR, cv::COLOR_BGR2GRAY);
 H=imgR.size().height;
 W=imgR.size().width;
 SetData(imgR.size().width, imgR.size().height);
 for(int row=0;row<H;row++)
  for(int col=0;col<W;col++)
   data[row*W+col]= ( imgR.data+row*W)[col];
 img.release();
 imgR.release();
}

void Image::WriteData(const std::string& nf) {
 cv::Mat imgW(H,W,CV_8U);
 for(int row=0;row<H;row++)
  for(int col=0;col<W;col++)
   (( imgW.data+row*W)[col])=data[row*W+col];
 if(!cv::imwrite(nf,imgW)) 
  cout<<"Cannot save the image"<<endl;
  imgW.release();
}

} // namespace core
} // namespace rtlp
