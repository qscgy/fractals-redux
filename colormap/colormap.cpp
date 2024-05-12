#include "colormap.h"
#include <iostream>

long interp_color(const colormap& cmap, const double val, const double min, const double max){
  const long* cmap_vals=cmap.cmap_vals;
  double nvals=cmap.nvals;
  if(val<min || val>max){
    throw std::invalid_argument("val must be between min and max");
  }

  double sval = (val - min)/(max - min);

  double lower, upper, subscaled;
  double rgb [3];
  unsigned char l, u;
  for(int i=0; i<nvals-1; i++){
    lower = ((double)i)/nvals;
    upper = ((double)(i+1))/nvals;
    if(sval>=lower && sval<upper){
      subscaled = (sval-lower)/(upper-lower);
      for(int c=0; c<3; c++){
        l=(*(cmap_vals+i) >> (c * 8)) & 0xFF;
        u=(*(cmap_vals+i+1) >> (c * 8)) & 0xFF;
        // printf("c=%d  l=%d  u=%d\n", c, l, u);
        rgb[2-c]=((double)(u-l))*subscaled + ((double)l);
      }
      // double val = *(cmap_vals+i) + subscaled*(*(cmap_vals+i+1)-*(cmap_vals+i));
      long retval = (std::lround(rgb[0]) << 16) + (std::lround(rgb[1]) << 8) + std::lround(rgb[2]);
      return retval;
    }
  }
  // printf("color: %x\n", *(cmap_vals + (((int)nvals)-1)));
  return *(cmap_vals + (((int)nvals)-1));
}

long interp_color(const colormap& cmap, const double val){
  return interp_color(cmap, val, 0, 1);
}
