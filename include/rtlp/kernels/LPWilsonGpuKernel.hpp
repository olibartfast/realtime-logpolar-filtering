#pragma once

namespace rtlp {
namespace kernels {

__device__ void kernelMask(int R, int S, int u, int v, float *mask, int rmax, int radius, float x, float y, float sigma);
__global__ void radiusKernel(float p0,float a, int R, int *radiusArray_d, float *sigmaArray_d);
__global__ void IMGsetKernel(float* IMG, int* d, int W, int H, int rmax);
__global__ void gaussFilterKernel(int R,int S, int W, int H, int rmax, int umaxfidx, float p0, int *cort,
	int *pradius, float *psigma, float *x, float *y, float *IMG);
__global__ void antiTransformKernel1(int R,int S, int W, int H, int radiusmax, int umaxfidx, float *IMG, float *NOR,
	float *x, float *y, int *pradius, float *psigma, int *data);
__global__ void	antiTransformKernel2(int R, int S, int W,int H,int umaxfidx,int rmax,int *ret,
										float *e,  int *pradius, float *IMG, float *NOR);

} // namespace kernels
} // namespace rtlp
