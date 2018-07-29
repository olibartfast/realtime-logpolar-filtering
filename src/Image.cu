#include "Image.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

Image::Image(){data=NULL;}


Image::~Image(){
 delete [] data;
 if(ret!=NULL)
 cudaFree(ret);
 if(cort!=NULL)
 cudaFree(cort);
}


void Image::SetData(int Wi,int He){
 if (data!=NULL)	
  delete [] data;
 data=new int[Wi*He];

}


void Image::SetData(int Wi,int He, int* pnt){
 if (data!=NULL)	
 delete [] data;
 W=Wi; H=He;
 data=new int[Wi*He];
 for(int row=0; row<He; row++)
  for(int col=0; col<Wi; col++)
   data[row*Wi+col]=pnt[row*Wi+col];
}



void Image::SetDataGpuR(int *d){
 if(ret!=NULL)
 cudaFree(ret);
 cudaMalloc((void**)&ret, W*H*sizeof(int));
 cudaMemcpy(ret, d, W*H*sizeof(int), cudaMemcpyHostToDevice);
}

void Image::SetDataGpuC(int R, int S){
 if(cort!=NULL)
 cudaFree(cort);
 cudaMalloc((void**)&cort, R*S*sizeof(int));
}

void Image::ReadData(string nf)
{
 cv::Mat img=cv::imread(nf);
 if (!img.data){ 
  cout<<"Non e' possibile leggere l'immagine"<<endl;
  return;
 }	
 cv::Mat imgR;
 cvtColor(img, imgR, CV_RGB2GRAY);
 H=imgR.size().height;
 W=imgR.size().width;
 SetData(imgR.size().width, imgR.size().height);
 for(int row=0;row<H;row++)
  for(int col=0;col<W;col++)
   data[row*W+col]= ( imgR.data+row*W)[col];
 img.release();
 imgR.release();
}

void Image::WriteData(string nf){
 cv::Mat imgW(H,W,CV_8U);
 for(int row=0;row<H;row++)
  for(int col=0;col<W;col++)
   (( imgW.data+row*W)[col])=data[row*W+col];
 if(!cv::imwrite(nf,imgW)) 
  cout<<"Non e' possibile salvare l'immagine"<<endl;
  imgW.release();
}
