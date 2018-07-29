#include "LogPolar.h"


LogPolar::LogPolar(Image *i, bool inv){
 this->inv=inv;
 imgfilter=i;
 W=i->GetW(); H=i->GetH();
 R=W/4;

//Per visualizzazione con la webcam il paramentro S (rappresentato sull'asse delle ordinate ) 
//non deve essere impostato ad un valore maggiore di quello massimo supportato dalla webcam
//(in questo caso 480 pixel)
 y0=H/2-1,x0=W/2-1; //posizione del centro
 p0=1;
 pmax=0.5*(min(W,H));
 a=exp(log(pmax/p0)/(float)R);
 S=static_cast<int>(2*M_PI/(a-1));
 q=(float)S/(2*M_PI);
 imgfilter->SetDataGpuC(R,S);
}
