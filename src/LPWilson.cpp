#include "LPWilson.h"

namespace rtlp {

//Constructor and destructor
LPWilson::LPWilson(Image *i, bool inv):LPBilinear(i,inv){}
LPWilson::~LPWilson(){
 for(int i=0; i<R*S; i++)
 {
  if(kernelpnt[i].kernelmatrix!=NULL)
   delete [] kernelpnt[i].kernelmatrix;
  }
 delete [] kernelpnt;
}

//Method that launches image processing
void LPWilson::process()
{
 create_map();
 to_cortical();
 if(inv)
  to_cartesian();
}

//Map creation
void LPWilson::create_map(){
 float pc, oc;
 kernelpnt=new kernel[R*S];
 umaxfidx=0;
 bool umaxf=false;
 for(int u=0; u<R; u++)
 {
  if((p0*(a-1)*pow(a,u-1)>1)&&(umaxf==false))
  {
   umaxfidx=u;
   umaxf=true;
  }
 }
 for (int v=0; v<S; v++)
 {
  for (int u=0; u<R; u++)
  {
   pc=(p0*pow(a,u));
   kernelpnt[v*R+u].radius=0;
   oc=((float)(v)/q);
   //Cartesian coordinate of the receptive field
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
    o=(atan(y/x));
   else o=(atan(y/x))+(M_PI);
   if(o<0)
	o+=(2*M_PI);
   e[i*W+j]=log(p/p0)/log(a);
   n[i*W+j]=q*o;
  }
 }
}
for(int i=0; i<R*S; i++)
 kernelpnt[i].kernelmatrix=NULL;
int xw,yw;
for(int v=0; v<S; v++)
 for(int u=umaxfidx; u<R; u++)
 {
  float sigma=(p0*(a-1)*pow(a,u-1))/3.0;
  int radius=(int)floor(3*sigma+0.5);
  kernelpnt[v*R+u].radius=radius;
  kernelpnt[v*R+u].kernelmatrix=new float[(2*radius+1)];
  xw=floor(xc[v*R+u]);
  yw=floor(yc[v*R+u]);
  float dx=xc[v*R+u]-xw;
  float dy=yc[v*R+u]-yw;
  float sum=0;
  float r=sqrt(dx*dx+dy*dy);
  for(int i=0; i<2*radius+1; i++)
  {
   (kernelpnt[v*R+u].kernelmatrix)[i]=exp(-(pow(i-radius-r, 2)/(2*sigma*sigma)));
   sum+=(kernelpnt[v*R+u].kernelmatrix)[i];
   }
  for(int i=0; i<2*radius+1; i++)
   (kernelpnt[v*R+u].kernelmatrix)[i]/=sum;
 }
}

//Passage to the cortical plane
void LPWilson::to_cortical(){
 int *cort=new int[S*R];
 for (int j=0; j<(S*R); j++)
  cort[j]=0;
 interp(cort, xc,yc,W,H,R,S,true);
 int radiusmax=kernelpnt[R-1].radius;
 float *IMG=new float[(W+2*radiusmax+1)*(H+2*radiusmax+1)];
 for(int j=0; j<(H+2*radiusmax+1)*(W+2*radiusmax+1); j++)
  IMG[j]=0;
 for(int j=0; j<H; j++)
  for(int i=0; i<W; i++)
   IMG[(W+2*radiusmax+1)*(j+radiusmax)+i+radiusmax]=imgfilter->Get(i,j);
 int xw,yw;
 for(int v=0; v<S; v++)
  for(int u=umaxfidx; u<R; u++)
   {
	xw=(int)floor(xc[v*R+u]);yw=(int)floor(yc[v*R+u]);
	int radius=kernelpnt[v*R+u].radius;
	float tmp=0;
	for(int rf=0; rf<(2*radius+1); rf++)
	{
	 for(int cf=0; cf<(2*radius+1); cf++)
	 {
	  tmp+=IMG[(W+2*radiusmax+1)*((rf-radius)+yw+radiusmax)+((cf-radius)+xw+radiusmax)]*
		   (kernelpnt[v*R+u]).kernelmatrix[cf]*
		   (kernelpnt[v*R+u]).kernelmatrix[rf];
	  }
	}
	cort[v*R+u]=(int) floor(tmp+0.5);
 }
 imgfilter->SetData(R,S,cort);
 delete [] cort;
 delete [] IMG;
}

//Inverse transformation for passage to the cartesian plane
void LPWilson::to_cartesian(){
 int *ret= new int [W*H];
 for (int j=0; j<(W*H); j++)
  ret[j]=0;
 interp(ret,e,n,R,S,W,H,false);
 int radiusmax=kernelpnt[(R-1)].radius;
 float *IMG=new float[(H+2*radiusmax+1)*(W+2*radiusmax+1)];
 float *NOR=new float[(H+2*radiusmax+1)*(W+2*radiusmax+1)];
 for(int i=0; i<((H+2*radiusmax+1)*(W+2*radiusmax+1)); i++)
 {
  IMG[i]=0;
  NOR[i]=0;
  }
 int xw,yw;
 for(int v=0; v<S; v++)
 for(int u=umaxfidx; u<R; u++)
 {
  xw=floor(xc[v*R+u]);
  yw=floor(yc[v*R+u]);
  int radius=kernelpnt[v*R+u].radius;
  for(int j=0; j<(2*radius+1); j++)
  {
   for(int i=0; i<(2*radius+1); i++)
   {
	int ind=(W+2*radiusmax+1)*((j-radius)+yw+radiusmax)+(i-radius)+xw+radiusmax;
	IMG[ind]+=kernelpnt[v*R+u].kernelmatrix[j]*kernelpnt[v*R+u].kernelmatrix[i]*imgfilter->Get(u, v);
	NOR[ind]+=(kernelpnt[v*R+u].kernelmatrix[j])*(kernelpnt[v*R+u].kernelmatrix[i]);
	}
   }
 }
 for(int i=0; i<((H+2*radiusmax+1)*(W+2*radiusmax+1)); i++)
  IMG[i]/=NOR[i];
 for(int j=radiusmax; j<H+radiusmax; j++)
  for(int i=radiusmax; i<W+radiusmax; i++)
  {
   int csi=(int) floor(e[W*(j-radiusmax)+i-radiusmax]);
   if((csi>=(umaxfidx-(kernelpnt[umaxfidx]).radius))&&(csi<R))
    ret[W*(j-radiusmax)+i-radiusmax]=(int) floor(IMG[(W+2*radiusmax+1)*j+i]+0.5);
  }
 imgfilter->SetData(W,H,ret);
 delete [] ret;
 delete [] IMG;
 delete [] NOR;
}

} // namespace rtlp
