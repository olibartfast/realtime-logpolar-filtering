#ifndef LOGPOLAR_H
#define LOGPOLAR_H

#include "Image.h"

#define BLOCKSZ 32

class LogPolar{
protected:
 Image *imgfilter;
 int W,H;
 float p, o; // rho e theta
 float p0; //raggio minimo
 float pmax,a,q;
 int R, S; //dimensioni dell'immagine logpolare R (colonna) S (riga)
 int x0,y0;
 bool inv;

public:
 LogPolar(Image *i, bool inv);
 ~LogPolar(){}
 virtual void create_map()=0;
 virtual void to_cortical()=0;
 virtual void to_cartesian()=0;
 virtual void process()=0;
};



#endif

