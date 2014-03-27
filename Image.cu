#include "Image.h"


 int* Image::GetGpuRPnt()
 {
	 return ret;
 }
 int* Image::GetGpuCPnt()
 {
	 return cort;
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


