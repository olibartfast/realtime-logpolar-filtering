#pragma once

#include "rtlp/core/Image.hpp"

#define BLOCKSZ 32

namespace rtlp {
namespace processing {

class LogPolar{
protected:
 rtlp::core::Image *imgfilter;
 int W,H;
 float p, o; // rho and theta
 float p0; //minimum radius
 float pmax,a,q;
 int R, S; //log-polar image dimensions R (column) S (row)
 int x0,y0;
 bool inv;

public:
 LogPolar(rtlp::core::Image *i, bool inv);
 ~LogPolar(){}
 virtual void create_map()=0;
 virtual void to_cortical()=0;
 virtual void to_cartesian()=0;
 virtual void process()=0;
};

} // namespace processing
} // namespace rtlp

