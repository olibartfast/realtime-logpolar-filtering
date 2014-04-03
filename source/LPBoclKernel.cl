#define float PI=3.141592653589793;

__kernel void createCorticalMapKernel(int x0, int y0, float a, float q, float p0,  __global float *xc, __global float *yc, int R, int S)
{
 int u=global_id(0);
 int v=global_id(1);
 if( v<S && u<R)
 {
  float pc, oc;
  pc=(p0*pow(a,u));
  oc=((float)(v)/q);
  //coordinata cartesiana del campo recettivo
  yc[v*R+u]=pc*sin((oc))+y0;
  xc[v*R+u]=pc*cos((oc))+x0;
 }
}


__kernel void createRetinalMapKernel(int x0, int y0, float a, float q, float p0, __global float *e, __global float *n, int W, int H)
{
 int j=global_id(0);
 int i=global_id(1);	
 if (i<H && j<W)
 {
  float p,o;
  float y=i-y0;
  float x=j-x0;
  p=floor(sqrt(x*x+y*y));
  if(x>=0)
   o=(atan(y/x));
  else o=(atan(y/x))+(PI);
  if(o<0)
   o+=(2*PI);
  e[i*W+j]=log(p/p0)/log(a);
  n[i*W+j]=q*o;
 }
}


__kernel void interpKernel(__global int *out, __global float *x, __global float *y, int Win, int Hin, int Wout, int Hout, bool way, __global int *d)
{
 int col=global_id(0);
 int row=global_id(1);
 if(row<Hout && col<Wout){
  int xw=floor(x[row*Wout+col]);
  int yw=floor(y[row*Wout+col]);
  float  xs=fabs(x[row*Wout+col]-floor(x[row*Wout+col])), ys=fabs(y[row*Wout+col]-ceil(y[row*Wout+col]));
    if ( (xw>=0 && xw<Win-1) && (yw>=0 && yw<Hin-1)) 
				out[row*Wout+col]=floor((1.0-xs)*(1.0-ys)*d[yw*Win+xw]+
									xs*(1.0-ys)*d[yw*Win+xw+1]+
									(1.0-xs)*ys*d[(yw+1)*Win+xw]+
									xs*ys*d[(yw+1)*Win+xw+1]+0.5);
			else if ( (xw>=0 && xw==Win-1) && (yw>=0 && yw<Hin-1)) 
				out[row*Wout+col]=floor((1.0-xs)*(1.0-ys)*d[yw*Win+xw]+
									xs*(1.0-ys)*d[yw*Win+xw-1]+
									(1.0-xs)*ys*d[(yw+1)*Win+xw]+
									xs*ys*d[(yw+1)*Win+xw-1]+0.5);
			else if ( (xw>=0 && xw<Win-1) && (yw>=0 && yw==Hin-1)) 
				if(way)
					out[row*Wout+col]=floor((1.0-xs)*(1.0-ys)*d[yw*Win+xw]+
									xs*(1.0-ys)*d[yw*Win+xw+1]+
									(1.0-xs)*ys*d[(yw-1)*Win+xw]+
									xs*ys*d[(yw-1)*Win+xw+1]+0.5);
				else
					out[row*Wout+col]=floor((1.0-xs)*(1.0-ys)*d[yw*Win+xw]+
									xs*(1.0-ys)*d[yw*Win+xw+1]+
									(1.0-xs)*ys*d[0*Win+xw]+
									xs*ys*d[0*Win+xw+1]+0.5);
			else if ( (xw>=0 && xw==Win-1) && (yw>=0 && yw==Hin-1)) 
				if(way)
					out[row*Wout+col]=floor((1.0-xs)*(1.0-ys)*d[yw*Win+xw]+
									xs*(1.0-ys)*d[yw*Win+xw-1]+
									(1.0-xs)*ys*d[(yw-1)*Win+xw]+
									xs*ys*d[(yw-1)*Win+xw-1]+0.5);
				else
					out[row*Wout+col]=floor((1.0-xs)*(1.0-ys)*d[yw*Win+xw]+
									xs*(1.0-ys)*d[yw*Win+xw-1]+
									(1.0-xs)*ys*d[0*Win+xw]+
									xs*ys*d[0*Win+xw-1]+0.5);
			else 
				out[row*Wout+col]=0;
	}
}

