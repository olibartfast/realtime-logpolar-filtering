#pragma once

#include "LPBilinear.hpp"

namespace rtlp {
namespace processing {

struct kernel { float *kernelmatrix; int radius;};

class LPWilson : public LPBilinear{
	
	kernel *kernelpnt;
	int umaxfidx;

public:
	LPWilson(rtlp::core::Image *i, bool inv);
	~LPWilson();

	void create_map();
	void to_cortical();
	void to_cartesian();
	void process();


};

} // namespace processing
} // namespace rtlp
