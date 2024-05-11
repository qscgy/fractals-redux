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

long interp_color(const int cmap, const double val, const double min, const double max);
long interp_color(const int cmap, const double val);