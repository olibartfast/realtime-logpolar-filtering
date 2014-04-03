#ifndef LPBILINEAR_H
#define LPBILINEAR_H

#include "LogPolar.h"


class LPBilinear : public LogPolar{
protected:
	float *e, *n; // epsilon (idx colonna) e eta (idx riga)
	float  *xc, *yc; //coordinate cartesiane x (idx colonna) e y (idx riga)
public:
	LPBilinear(Image *i, bool inv);
	~LPBilinear();
	
	void interp(int* out, float *x, float *y, int Win, int Hin, int Wout, int Hout, bool way);
	void create_map();
	void to_cortical();
	void to_cartesian();
	void process();
};


#endif
