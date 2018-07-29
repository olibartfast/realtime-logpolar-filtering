#ifndef LPBILINEARGPUKERNEL_H
#define LPBILINEARGPUKERNEL_H

__global__ void createCorticalMapKernel(int x0, int y0, float a, float q, float p0,  float *xc, float *yc, int R, int S);
__global__ void createRetinalMapKernel(int x0, int y0, float a, float q, float p0, float *e, float *n, int W, int H);
__global__ void interpKernel(int *out, float *x, float *y, int Win, int Hin, int Wout, int Hout, bool way, int *d);

#endif
