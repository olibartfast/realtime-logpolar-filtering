#pragma once

#include "LogPolar.h"

namespace rtlp {
namespace processing {

class LPBilinearGpu : public LogPolar{
 protected:
  float *xc_d, *yc_d;
  float *e_d, *n_d;
 public:
  LPBilinearGpu(Image *i, bool inv);
  ~LPBilinearGpu();
  void create_map();
  void to_cortical();
  void to_cartesian();
  void process();



};

} // namespace processing
} // namespace rtlp
