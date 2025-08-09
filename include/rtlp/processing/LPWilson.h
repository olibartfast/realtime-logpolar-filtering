#pragma once

#include "LPBilinear.h"

namespace rtlp {
namespace processing {

struct kernel { float *kernelmatrix; int radius;};

class LPWilson : public LPBilinear{
	
	kernel *kernelpnt;
	int umaxfidx;

public:
	LPWilson(Image *i, bool inv);
	~LPWilson();

	void create_map();
	void to_cortical();
	void to_cartesian();
	void process();


};

} // namespace processing
} // namespace rtlp
