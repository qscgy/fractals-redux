#include "colormap.h"
#include <iostream>
#include <cmath>
#include <string>

Colormap::Colormap(){};

Colormap::Colormap(const std::string &cmapStr)
{
  cmap_vals = palettes.at(cmapStr).data();
  nvals = (int)(palettes.at(cmapStr).size());

  hsv_vals = new double *[nvals];
  double rgb_n[3];
  for (int i = 0; i < nvals; i++)
  {
    for (int c = 0; c < 3; c++)
    {
      rgb_n[2 - c] = (cmap_vals[i] >> (c * 8)) & 0xFF;
    }
    // hsv_vals[i]
    hsv_vals[i] = new double[3];
    rgb2hsv(rgb_n, hsv_vals[i]);
    // hsv_vals[i] = hsv[i];
  }
  // hsv_vals = &hsv;
}

long interp_color(const Colormap &cmap, const double val, const double min, const double max)
{
  const long *cmap_vals = cmap.cmap_vals;
  double nvals = cmap.nvals;
  if (val < min || val > max)
  {
    throw std::invalid_argument("val must be between min and max");
  }

  double sval = (val - min) / (max - min);

  double lower, upper, subscaled;
  double rgb[3], hsv[3];
  unsigned char l, u;
  double lf, uf;
  double *ld;
  double *ud;
  for (int i = 0; i < nvals - 1; i++)
  {
    lower = ((double)i) / nvals;
    upper = ((double)(i + 1)) / nvals;
    if (sval >= lower && sval < upper)
    {
      subscaled = (sval - lower) / (upper - lower);
      for (int c = 0; c < 3; c++)
      {
        l = (*(cmap_vals + i) >> (c * 8)) & 0xFF;
        u = (*(cmap_vals + i + 1) >> (c * 8)) & 0xFF;
        // rgb[2 - c] = ((double)(u - l)) * subscaled + ((double)l);
        lf = rgb2linear(l);
        uf = rgb2linear(u);
        rgb[2 - c] = (uf - lf) * subscaled + lf;
      }

      // ld = *(cmap.hsv_vals + i);
      // ud = *(cmap.hsv_vals + i + 1);
      // hsv[0] = ((*ud) - (*ld))*subscaled + (*ld);
      // hsv[1] = (*(ud + 1) - *(ld + 1)) * subscaled + *(ld + 1);
      // hsv[2] = (*(ud + 2) - *(ld + 2)) * subscaled + *(ld + 2);
      // hsv2rgb(hsv, rgb);

      // long retval = (std::lround(rgb[0]) << 16) + (std::lround(rgb[1]) << 8) + std::lround(rgb[2]);
      long retval = (linear2rgb(rgb[0]) << 16) + (linear2rgb(rgb[1]) << 8) + linear2rgb(rgb[2]);
      return retval;
    }
  }
  return *(cmap_vals + (((int)nvals) - 1));
}

/**
 * Convert an RGB color to HSV.
 * @param rgb double[3] representing RGB color, normalized from 0 to 1.
 * @param hsv double* pointing to array of length 3, in which to store the HSV color output
 */
void rgb2hsv(const double rgb[3], double hsv[3])
{
  double cmax = std::max(rgb[0], std::max(rgb[1], rgb[2]));
  double cmin = std::min(rgb[0], std::min(rgb[1], rgb[2]));
  if (cmax == cmin)
  {
    hsv[0] = 0;
    hsv[1] = 0;
    hsv[2] = cmax;
    return;
  }
  double s = (cmax - cmin) / cmax;
  double rc = (cmax - rgb[0]) / (cmax - cmin);
  double gc = (cmax - rgb[1]) / (cmax - cmin);
  double bc = (cmax - rgb[2]) / (cmax - cmin);
  double h;
  if (cmax == rgb[0])
  {
    h = bc - gc;
  }
  else if (cmax == rgb[1])
  {
    h = 2.0 + rc - bc;
  }
  else
  {
    h = 4.0 + gc - rc;
  }
  h = std::fmod(h / 6.0, 1.0);
  hsv[0] = h;
  hsv[1] = s;
  hsv[2] = cmax;
}

void hsv2rgb(const double hsv[3], double rgb[3])
{
  double h = fmod(hsv[0] * 360.0, 360.0);
  double c = hsv[1] * hsv[2];
  double x = c * (1 - abs(fmod(h / 60, 2.0) - 1));
  double m = hsv[2] - c;
  if (h < 60.0)
  {
    rgb[0] = c + m;
    rgb[1] = x + m;
    rgb[2] = m;
  }
  else if (h < 120.0)
  {
    rgb[0] = x + m;
    rgb[1] = c + m;
    rgb[2] = m;
  }
  else if (h < 180.0)
  {
    rgb[0] = m;
    rgb[1] = c + m;
    rgb[2] = x + m;
  }
  else if (h < 240.0)
  {
    rgb[0] = m;
    rgb[1] = x + m;
    rgb[2] = c + m;
  }
  else if (h < 300)
  {
    rgb[0] = x + m;
    rgb[1] = m;
    rgb[2] = c + m;
  }
  else
  {
    rgb[0] = c + m;
    rgb[1] = m;
    rgb[2] = x + m;
  }
}

double rgb2linear(unsigned char channel)
{
  double s = channel / 255.0;
  return s <= 0.04045 ? s / 12.92 : pow((s + 0.055) / 1.055, 2.4);
}

long linear2rgb(double linear)
{
  double s = linear <= 0.0031308 ? linear * 12.92 : 1.055 * pow(linear, 1.0 / 2.4) - 0.055;
  return std::lround(s * 255);
}

long interp_color(const Colormap &cmap, const double val)
{
  return interp_color(cmap, val, 0, 1);
}
