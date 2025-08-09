#include "LPBilinearGpuKernel.h"
#include "LPWilsonGpu.h"		
#include "LPWilsonGpuKernel.cu"

namespace rtlp {

LPWilsonGpu::LPWilsonGpu(Image *i, bool inv):LPBilinearGpu(i,inv){}

LPWilsonGpu::~LPWilsonGpu(){
 cudaFree(radiusArray_d);
 cudaFree(sigmaArray_d);
 cudaFree(IMG_d);
}

void LPWilsonGpu::process()
{
 create_map();
 to_cortical();
 if(inv)
  to_cartesian();
}


void LPWilsonGpu::create_map(){
 
 cudaMalloc((void**)&xc_d, R*S*sizeof(float));
 cudaMalloc((void**)&yc_d, R*S*sizeof(float));
 dim3 dimBlock(BLOCKSZ, BLOCKSZ);
 dim3 dimCGrid(R/dimBlock.x+1, S/dimBlock.y+1);

 bool uMaxFoveaCheck=false;
 for(int u=0; uMaxFoveaCheck==false; u++)
 {
  if(p0*(a-1)*pow(a,u-1)>1)
  {
   uMaxFovea=u;
   uMaxFoveaCheck=true;
  }
 }

 cudaMalloc((void**)&radiusArray_d, R*sizeof(int));
 cudaMalloc((void**)&sigmaArray_d, R*sizeof(float));

 radiusKernel<<<R/512+1,512>>>(p0,a,R, radiusArray_d, sigmaArray_d);

 cudaMemcpy(&rfMaxRadius,radiusArray_d+(R-1),sizeof(int),cudaMemcpyDeviceToHost);



 createCorticalMapKernel<<<dimCGrid, dimBlock>>>(x0,y0,a,q,p0, xc_d,yc_d,R,S);



 if (inv)
 {
  cudaMalloc((void**)&e_d, W*H*sizeof(float));
  cudaMalloc((void**)&n_d, W*H*sizeof(float));
  dim3 dimRGrid(W/dimBlock.x+1, H/dimBlock.y+1);
  createRetinalMapKernel<<<dimRGrid, dimBlock>>>(x0,y0,a,q,p0,e_d,n_d,W,H);
 }
}
	
//----------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------
void LPWilsonGpu::to_cortical()
{
 int *cort=(int*)malloc(R*S*sizeof(int));

  dim3 dimBlock(BLOCKSZ, BLOCKSZ);
  dim3 dimRSGrid(R/dimBlock.x+1, S/dimBlock.y+1);
 
 interpKernel<<<dimRSGrid, dimBlock>>>(imgfilter->GetGpuCPnt(), xc_d, yc_d, W, H, R, S,true, imgfilter->GetGpuRPnt());

 cudaMalloc((void**)&IMG_d, (W+2*rfMaxRadius+1)*(H+2*rfMaxRadius+1)*sizeof(float));
 dim3 dimBlockIMG(BLOCKSZ,BLOCKSZ);
 dim3 dimGridIMG(W/dimBlockIMG.x+1, H/dimBlockIMG.y+1);
 IMGsetKernel<<<dimGridIMG,dimBlockIMG>>>(IMG_d, imgfilter->GetGpuRPnt(), W,H,rfMaxRadius);

 dim3 dimGrid(R/dimBlock.x+1, S/dimBlock.y+1);


 gaussFilterKernel<<<dimGrid, dimBlock>>>(R,S,W,H,rfMaxRadius,uMaxFovea,p0,imgfilter->GetGpuCPnt(),
		 	 	 	 	 	 	 	 	 radiusArray_d , sigmaArray_d, xc_d, yc_d, IMG_d);

 cudaMemcpy(cort, imgfilter->GetGpuCPnt(), R*S*sizeof(int), cudaMemcpyDeviceToHost);
 imgfilter->SetData(R,S,cort);
 free(cort);


}


void LPWilsonGpu::to_cartesian()
{
 int *ret= (int*)malloc(W*H*sizeof(int));

  dim3 dimBlock(BLOCKSZ, BLOCKSZ);
  dim3 dimWHGrid(W/dimBlock.x+1, H/dimBlock.y+1);
 
 interpKernel<<<dimWHGrid, dimBlock>>>(imgfilter->GetGpuRPnt(), e_d, n_d, R, S, W, H,false, imgfilter->GetGpuCPnt());

 float *NOR_d;
 cudaMalloc((void**)&NOR_d, (H+2*rfMaxRadius+1)*(W+2*rfMaxRadius+1)*sizeof(float));
 cudaMemset(NOR_d, 0, (H+2*rfMaxRadius+1)*(W+2*rfMaxRadius+1)*sizeof(float));
 cudaMemset(IMG_d, 0, (H+2*rfMaxRadius+1)*(W+2*rfMaxRadius+1)*sizeof(float));


 dim3 dimGrid(R/dimBlock.x+1, S/dimBlock.y+1);

 antiTransformKernel1<<<dimGrid, dimBlock>>>(R,S,W,H,rfMaxRadius,uMaxFovea,IMG_d,NOR_d,
		 	 	 	 	xc_d, yc_d, radiusArray_d, sigmaArray_d, imgfilter->GetGpuCPnt());

 dim3 dimGrid2((W+2*rfMaxRadius+1)/dimBlock.x+1, (H+2*rfMaxRadius+1)/dimBlock.y+1);
 antiTransformKernel2<<<dimGrid2,dimBlock>>>(R,S,W,H,uMaxFovea,rfMaxRadius,imgfilter->GetGpuRPnt(), e_d, radiusArray_d, IMG_d, NOR_d);

 cudaMemcpy(ret, imgfilter->GetGpuRPnt(), W*H*sizeof(int), cudaMemcpyDeviceToHost);

 imgfilter->SetData(W,H,ret);

 free(ret);
 cudaFree(NOR_d);

}

} // namespace rtlp
