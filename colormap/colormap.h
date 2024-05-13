#include <map>
#include <vector>
#include <string>
#pragma once

const double _PI = std::acos(-1.0);

const std::map<std::string, std::vector<long>> palettes {
  {
    "magma",
    std::vector<long> {0xffdb00, 0xffa904, 0xee7b06, 0xa12424, 0x400b0b}
  },
  {
    "grayscale",
    std::vector<long> {0x000000, 0x555555, 0xaaaaaa, 0xffffff}
  },
  {
    "rainbow",
    std::vector<long> {0xe81416, 0xffa500, 0xfaeb36, 0x79c314, 0x487de7, 0x4b369d, 0x70369d}
  },
  {
    "grape",
    std::vector<long> {0x522956, 0x7b395c, 0x9d505d, 0xb66d5d}
  },
  {
    "magma2",
    std::vector<long> {0x000004,0x51127c,0xb73779,0xfc8961,0xfcfdbf}
  },
  {
    "viridis",
    std::vector<long> {0x440154,0x3b528b,0x21918c,0x5ec962,0xfde725}
  },
  {
    "red2blue",
    std::vector<long> {0xFF1010, 0xE21122, 0xC51235, 0xA81347, 0x8C145A, 
                       0x6F156C, 0x52167F, 0x351791, 0x1919A4}
  },
  {
    "rainbow2",
    std::vector<long> {0x031B26,	0x094F17,	0x707A1C,	0xA53535,	0xBBC655,	0x85DB96,	0xB6DCED,	0xF7F2FF}
  }
};

struct colormap {
  const long* cmap_vals;
  int nvals;
};

class Colormap {
  public:
    const long* cmap_vals;
    double** hsv_vals;
    int nvals;
    Colormap(const std::string& cmapStr);
};

const int MAGMA = 0;
// const long magma [5] = {0x003f5c,0x58508d,0xbc5090,0xff6361,0xffa600};
const long magma [5] = {0xffdb00, 0xffa904, 0xee7b06, 0xa12424, 0x400b0b};
const int magma_len = 5;

const int GRAYSCALE = 1;
const long grayscale [4] = {0x000000, 0x555555, 0xaaaaaa, 0xffffff};
const int grayscale_len = 4;

const int RAINBOW = 2;
const long rainbow [7] = {0xe81416, 0xffa500, 0xfaeb36, 0x79c314, 0x487de7, 0x4b369d, 0x70369d};
const int rainbow_len = 7;

const int GRAPE = 3;
const long grape [4] = {0x522956, 0x7b395c, 0x9d505d, 0xb66d5d};
const int grape_len = 4;

const int MAGMA2 = 4;
const long magma2 [5] = {0x000004,0x51127c,0xb73779,0xfc8961,0xfcfdbf};
const int magma2_len = 5;

long interp_color(const colormap& cmap, const double val, const double min, const double max);
long interp_color(const colormap& cmap, const double val);

void rgb2hsv(const double rgb[3], double hsv[3]);
void hsv2rgb(const double hsv[3], double rgb[3]);
double rgb2linear(unsigned char channel);
long linear2rgb(double linear);