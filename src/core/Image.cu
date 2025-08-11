#include "rtlp/core/Image.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace rtlp {
namespace core {

Image::Image() : ret_(nullptr), cort_(nullptr) {}


Image::~Image(){
 // data is automatically cleaned up by unique_ptr
 if(ret_ != nullptr)
 cudaFree(ret_);
 if(cort_ != nullptr)
 cudaFree(cort_);
}


void Image::SetData(int Wi,int He){
 data_ = std::make_unique<int[]>(Wi*He);
 W = Wi;
 H = He;
}


void Image::SetData(int Wi,int He, int* pnt){
 W = Wi; 
 H = He;
 data_ = std::make_unique<int[]>(Wi*He);
 for(int row=0; row<He; row++)
  for(int col=0; col<Wi; col++)
   data_[row*Wi+col] = pnt[row*Wi+col];
}



void Image::SetDataGpuR(int *d){
 if(ret_ != nullptr)
 cudaFree(ret_);
 cudaMalloc((void**)&ret_, W*H*sizeof(int));
 cudaMemcpy(ret_, d, W*H*sizeof(int), cudaMemcpyHostToDevice);
}

void Image::SetDataGpuC(int R, int S){
 if(cort_ != nullptr)
 cudaFree(cort_);
 cudaMalloc((void**)&cort_, R*S*sizeof(int));
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
   data_[row*W+col]= ( imgR.data+row*W)[col];
 img.release();
 imgR.release();
}

void Image::WriteData(const std::string& nf) {
 cv::Mat imgW(H,W,CV_8U);
 for(int row=0;row<H;row++)
  for(int col=0;col<W;col++)
   (( imgW.data+row*W)[col])=data_[row*W+col];
 if(!cv::imwrite(nf,imgW)) 
  cout<<"Cannot save the image"<<endl;
  imgW.release();
}

} // namespace core
} // namespace rtlp
