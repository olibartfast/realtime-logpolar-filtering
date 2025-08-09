#pragma once

#include "LogPolar.h"

namespace rtlp {

class LPBilinear : public LogPolar{
protected:
	float *e, *n; // epsilon (column idx) and eta (row idx)
	float  *xc, *yc; //cartesian coordinates x (column idx) and y (row idx)
public:
	LPBilinear(Image *i, bool inv);
	~LPBilinear();
	
	void interp(int* out, float *x, float *y, int Win, int Hin, int Wout, int Hout, bool way);
	void create_map();
	void to_cortical();
	void to_cartesian();
	void process();
};

} // namespace rtlp
