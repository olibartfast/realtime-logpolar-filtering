namespace rtlp {

const __device__ float PI=3.141592653589793;

__global__ void createCorticalMapKernel(int x0, int y0, float a, float q, float p0,  float *xc, float *yc, int R, int S)
{
	int u=blockIdx.x*blockDim.x+threadIdx.x;
	int v=blockIdx.y*blockDim.y+threadIdx.y;


 
	if( v<S && u<R)
		{
			float pc, oc;
            pc=(p0*pow(a,u));
			oc=((float)(v)/q);
			//cartesian coordinate of the receptive field
			yc[v*R+u]=pc*sin((oc))+y0;
			xc[v*R+u]=pc*cos((oc))+x0;
		}
	
}


__global__ void createRetinalMapKernel(int x0, int y0, float a, float q, float p0, float *e, float *n, int W, int H)
{
	
	int j=blockIdx.x*blockDim.x+threadIdx.x;
	int i=blockIdx.y*blockDim.y+threadIdx.y;	

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


__global__ void interpKernel(int *out, float *x, float *y, int Win, int Hin, int Wout, int Hout, bool way, int *d)
{
	
	
	int col=blockIdx.x*blockDim.x+threadIdx.x;
	int row=blockIdx.y*blockDim.y+threadIdx.y;



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

} // namespace rtlp
