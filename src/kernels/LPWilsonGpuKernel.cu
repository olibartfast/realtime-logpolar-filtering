#include "../../include/rtlp/kernels/LPWilsonGpuKernel.h"

namespace rtlp {
namespace kernels {

__device__ void kernelMask(int R, int S, int u, int v, float *mask, int rmax, int radius, float x, float y, float sigma)
{
 float sum=0;
 float dx=x-floor(x);
 float dy=y-floor(y);
 float r=sqrt(dx*dx+dy*dy);

 for(int i=0; i<2*radius+1; i++)
  {
   mask[i]=exp(-(pow(i-radius-r, 2)/(2*sigma*sigma)));
		sum+=mask[i];
	}

	for(int i=0; i<2*radius+1; i++)
				mask[i]/=sum;

}

__global__ void radiusKernel(float p0,float a, int R, int *radiusArray_d, float *sigmaArray_d)
{
 int u=blockDim.x*blockIdx.x+threadIdx.x;
 if(u>=R) return;
 sigmaArray_d[u]=(p0*(a-1)*pow(a,u-1))/3.0;
 radiusArray_d[u]=(int)floor(3*sigmaArray_d[u]+0.5);
}


 __global__ void IMGsetKernel(float* IMG, int* d, int W, int H, int rmax)
 {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int j = blockDim.y*blockIdx.y+threadIdx.y;
  if (j<0 || j>=H  || i<0 || i>=W)
  return;
  IMG[(W+2*rmax+1)*(j+rmax)+i+rmax]=d[j*W+i];

 }


__global__ void gaussFilterKernel(int R,int S, int W, int H, int rmax, int umaxfidx, float p0, int *cort,
	int *pradius, float *psigma, float *x, float *y, float *IMG)
{
 int u=blockIdx.x*blockDim.x+threadIdx.x; //u ---> column index
 int v=blockIdx.y*blockDim.y+threadIdx.y; //v ---> row index
 int xw,yw;
 float w[32];
 if(u>=umaxfidx && u<R && v<S)
 {
  xw=(int)floor(x[v*R+u]);
  yw=(int)floor(y[v*R+u]);
  int radius=pradius[u];
  float tmp=0;
  kernelMask(R, S, u, v, w, rmax, radius, x[v*R+u], y[v*R+u], psigma[u]);
  for(int rf=0; rf<(2*radius+1); rf++)
  {
   for(int cf=0; cf<(2*radius+1); cf++)
   {
    float weight=w[cf]*w[rf];
    int idx=(W+2*rmax+1)*((rf-radius)+yw+rmax)+((cf-radius)+xw+rmax);
    tmp+=IMG[idx]*weight;

   }
   cort[v*R+u]=static_cast<int>(floor(tmp+0.5));
  }
 }
}

__global__ void antiTransformKernel1(int R,int S, int W, int H, int rmax, int umaxfidx, float *IMG, float *NOR,
	float *x, float *y, int *pradius, float *psigma, int *data)

{
 float w[32];
 int u=blockIdx.x*blockDim.x+threadIdx.x; //u ---> column index
 int v=blockIdx.y*blockDim.y+threadIdx.y; //v ---> row index
 if(u<umaxfidx || u>=R || v>=S)
  return;
 int xw=floor(x[v*R+u]);
 int yw=floor(y[v*R+u]);
 int radius=pradius[u];
 int dataval=data[v*R+u];
 kernelMask(R, S, u, v, w, rmax, radius, x[v*R+u], y[v*R+u], psigma[u]);

 for(int j=0; j<(2*radius+1); j++)
 {
  for(int i=0; i<(2*radius+1); i++)
  {
   float weight=w[i]*w[j];
   int ind=(W+2*rmax+1)*((j-radius)+yw+rmax)+(i-radius)+xw+rmax;
   atomicAdd(IMG+ind,weight*dataval);
   atomicAdd(NOR+ind,weight);

   }
  }
}




__global__ void	antiTransformKernel2(int R, int S, int W,int H,int umaxfidx,int rmax,int *ret,
										float *e,  int *pradius, float *IMG, float *NOR)
{
 int i=blockIdx.x*blockDim.x+threadIdx.x; //i ---> column index
 int j=blockIdx.y*blockDim.y+threadIdx.y; //j ---> row index
 if (j<rmax || j>=H+rmax  || i<rmax || i>=W+rmax)
  return;
 int idx=j*(W+2*rmax+1)+i;
 IMG[idx]/=NOR[idx];
 int csi=static_cast<int>(floor(e[(j-rmax)*(W)+i-rmax]));
 if( csi>= (umaxfidx-pradius[umaxfidx]) && csi<R )
  ret[W*(j-rmax)+i-rmax]=static_cast<int>(floor(IMG[(W+2*rmax+1)*j+i]+0.5));

}

} // namespace kernels
} // namespace rtlp
