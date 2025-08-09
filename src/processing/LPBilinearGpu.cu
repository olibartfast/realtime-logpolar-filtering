#include "rtlp/processing/LPBilinearGpu.hpp"
#include "../kernels/LPBilinearGpuKernel.cu"

namespace rtlp {
namespace processing {

LPBilinearGpu::LPBilinearGpu(rtlp::core::Image *i, bool inv):LogPolar(i,inv){}

LPBilinearGpu::~LPBilinearGpu()
{
 cudaFree(xc_d);
 cudaFree(yc_d);
 cudaFree(e_d);
 cudaFree(n_d);	

}

void LPBilinearGpu::process()
{
 create_map();
 to_cortical(); 
 if(inv)
  to_cartesian();
}


void LPBilinearGpu::create_map(){
 cudaMalloc((void**)&xc_d, R*S*sizeof(float));
 cudaMalloc((void**)&yc_d, R*S*sizeof(float));
 dim3 dimBlock(BLOCKSZ, BLOCKSZ);
 dim3 dimCGrid(R/dimBlock.x+1, S/dimBlock.y+1);
 
 kernels::createCorticalMapKernel<<<dimCGrid, dimBlock>>>(x0,y0,a,q,p0, xc_d,yc_d,R,S);


if (inv)
 {
  cudaMalloc((void**)&e_d, W*H*sizeof(float));
  cudaMalloc((void**)&n_d, W*H*sizeof(float));
  dim3 dimRGrid(W/dimBlock.x+1, H/dimBlock.y+1);
  kernels::createRetinalMapKernel<<<dimRGrid, dimBlock>>>(x0,y0,a,q,p0,e_d,n_d,W,H);
 }
}





void LPBilinearGpu::to_cortical(){
 int *cort=new int[R*S];

  dim3 dimBlock(BLOCKSZ, BLOCKSZ);
  dim3 dimGrid(R/dimBlock.x+1, S/dimBlock.y+1);
 
 kernels::interpKernel<<<dimGrid, dimBlock>>>(imgfilter->GetGpuCPnt(), xc_d, yc_d, W, H, R, S,true, imgfilter->GetGpuRPnt());

 cudaMemcpy(cort, imgfilter->GetGpuCPnt(), R*S*sizeof(int), cudaMemcpyDeviceToHost);

 imgfilter->SetData(R,S,cort);
 delete [] cort;
}

void LPBilinearGpu::to_cartesian(){
 int *ret= new int [W*H];

  dim3 dimBlock(BLOCKSZ, BLOCKSZ);
  dim3 dimGrid(W/dimBlock.x+1, H/dimBlock.y+1);
 
 kernels::interpKernel<<<dimGrid, dimBlock>>>(imgfilter->GetGpuRPnt(), e_d, n_d, R, S, W, H,false, imgfilter->GetGpuCPnt());

 cudaMemcpy(ret, imgfilter->GetGpuRPnt(), W*H*sizeof(int), cudaMemcpyDeviceToHost);


 imgfilter->SetData(W,H,ret);
 delete [] ret;
}

} // namespace processing
} // namespace rtlp
