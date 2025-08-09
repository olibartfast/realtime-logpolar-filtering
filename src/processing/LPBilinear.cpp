#include "rtlp/processing/LPBilinear.h"

namespace rtlp {
namespace processing {

LPBilinear::LPBilinear(Image *i, bool inv):LogPolar(i,inv)
{
	xc=new float[R*S];
	yc=new float[R*S];
	e=new float[W*H];
	n=new float[W*H];
}


LPBilinear::~LPBilinear()
{
		delete [] xc;
		delete [] yc;
		delete [] n;
		delete [] e;
}

void LPBilinear::create_map(){

	float pc, oc;

	for(int v=0; v<S; v++) 
	{
		oc=((float)(v)/q);
		for (int u=0; u<R; u++) 
		{
			pc=(p0*pow(a,u));
			//cartesian coordinate of the receptive field
			yc[v*R+u]=pc*sin((oc))+y0;
			xc[v*R+u]=pc*cos((oc))+x0;

		}
	}

       
       if(inv)
      {
	for (int i=0; i<H; i++)
	{
		float y=i-y0;
		for (int j=0; j<W; j++)
		{
			float x=j-x0;
			p=floor(sqrt(float(x*x+y*y)));
			if(x>=0)
				o=((atan(y/x)));
			else 
				o=((atan(y/x)))+(M_PI);
			if(o<0)
				o+=2*M_PI;
			e[i*W+j]=log(p/p0)/log(a);
			n[i*W+j]=q*o;
		}
	}
  }

}


void LPBilinear::interp(int *out, float *x, float *y, int Win, int Hin, int Wout, int Hout, bool way)
{

	for (int row=0; row<Hout; row++)
		for(int col=0; col<Wout; col++){

			int xw=(int) floor(x[row*Wout+col]);
			int yw=(int) floor(y[row*Wout+col]);

			float  xs=abs(x[row*Wout+col]-floor(x[row*Wout+col])), ys=abs(y[row*Wout+col]-ceil(y[row*Wout+col]));
			
			if ( (xw>=0 && xw<Win-1) && (yw>=0 && yw<Hin-1)) 
				out[row*Wout+col]=(int)floor((1.0-xs)*(1.0-ys)*imgfilter->Get(xw, yw)+
									xs*(1.0-ys)*imgfilter->Get(xw+1,yw)+
									(1.0-xs)*ys*imgfilter->Get(xw, yw+1)+
									xs*ys*imgfilter->Get(xw+1, yw+1)+0.5);
			else if ( (xw>=0 && xw==Win-1) && (yw>=0 && yw<Hin-1)) 
				out[row*Wout+col]=(int)floor((1.0-xs)*(1.0-ys)*imgfilter->Get(xw, yw)+
									xs*(1.0-ys)*imgfilter->Get(xw-1,yw)+
									(1.0-xs)*ys*imgfilter->Get(xw, yw+1)+
									xs*ys*imgfilter->Get(xw-1, yw+1)+0.5);
			else if ( (xw>=0 && xw<Win-1) && (yw>=0 && yw==Hin-1)) 
				if(way)
					out[row*Wout+col]=(int)floor((1.0-xs)*(1.0-ys)*imgfilter->Get(xw, yw)+
									xs*(1.0-ys)*imgfilter->Get(xw+1,yw)+
									(1.0-xs)*ys*imgfilter->Get(xw, yw-1)+
									xs*ys*imgfilter->Get(xw+1, yw-1)+0.5);
				else
					out[row*Wout+col]=(int)floor((1.0-xs)*(1.0-ys)*imgfilter->Get(xw, yw)+
									xs*(1.0-ys)*imgfilter->Get(xw+1,yw)+
									(1.0-xs)*ys*imgfilter->Get(xw, 0)+
									xs*ys*imgfilter->Get(xw+1, 0)+0.5);
			else if ( (xw>=0 && xw==Win-1) && (yw>=0 && yw==Hin-1)) 
				if(way)
					out[row*Wout+col]=(int)floor((1.0-xs)*(1.0-ys)*imgfilter->Get(xw, yw)+
									xs*(1.0-ys)*imgfilter->Get(xw-1,yw)+
									(1.0-xs)*ys*imgfilter->Get(xw, yw-1)+
									xs*ys*imgfilter->Get(xw-1, yw-1)+0.5);
				else
					out[row*Wout+col]=(int)floor((1.0-xs)*(1.0-ys)*imgfilter->Get(xw, yw)+
									xs*(1.0-ys)*imgfilter->Get(xw-1,yw)+
									(1.0-xs)*ys*imgfilter->Get(xw, 0)+
									xs*ys*imgfilter->Get(xw-1, 0)+0.5);
			else out[row*Wout+col]=0;

	}
}




void LPBilinear::to_cortical(){

	int *cort=new int[R*S];
	for (int j=0; j<(R*S); j++)
		cort[j]=0;
	interp(cort,xc,yc,W,H,R,S,true);
	imgfilter->SetData(R,S,cort);
	delete [] cort;
}

void LPBilinear::to_cartesian(){

	int *ret= new int [W*H];
	for (int j=0; j<(W*H); j++)
		ret[j]=0;
	interp(ret,e,n,R,S,W,H,false);
	imgfilter->SetData(W,H,ret);
	delete [] ret;
}


void LPBilinear::process()
{
	create_map();
	to_cortical();
	if(inv)
		to_cartesian();
}

} // namespace processing
} // namespace rtlp
