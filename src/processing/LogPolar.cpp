#include "rtlp/processing/LogPolar.h"

namespace rtlp {
namespace processing {

LogPolar::LogPolar(rtlp::core::Image *i, bool inv){
 this->inv=inv;
 imgfilter=i;
 W=i->GetW(); H=i->GetH();
 R=W/4;

//For webcam visualization the parameter S (represented on the ordinate axis) 
//must not be set to a value greater than the maximum supported by the webcam
//(in this case 480 pixels)
 y0=H/2-1,x0=W/2-1; //center position
 p0=1;
 pmax=0.5*(min(W,H));
 a=exp(log(pmax/p0)/(float)R);
 S=static_cast<int>(2*M_PI/(a-1));
 q=(float)S/(2*M_PI);
 imgfilter->SetDataGpuC(R,S);
}

} // namespace processing
} // namespace rtlp
