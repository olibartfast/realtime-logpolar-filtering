#ifndef LPWILSONGPU_H
#define LPWILSONGPU_H

#include "LPBilinearGpu.h"

class LPWilsonGpu : public LPBilinearGpu{
 int *radiusArray_d;
 float *sigmaArray_d;
 int rfMaxRadius;
 int rfMaxRadius_d;
 int uMaxFovea;
 float *IMG_d;

 public:
  LPWilsonGpu(Image *i, bool inv);
  ~LPWilsonGpu();
  void create_map();
  void to_cortical();
  void to_cartesian();
  void process();
};
#endif
