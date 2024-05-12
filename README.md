# Fractal generator
This is a simple program to generate images of fractals. Right now it allows for Mandelbrot, Julia, and burning ship fractals, but I plan to add more.

## Requirements
- OpenCV for C++

## Installation
```
git clone https://github.com/qscgy/fractals-redux
cd fractals-redux
mkdir build
cd build
```
Now build the program:
```
cmake ..
make
```
## Using the fractal generator
To use this software, run it from the command line like so:
```
$ ./Fractal <path to config file> <optional directory to save to on exit>
```
The config file is a list of key-value pairs, one per line, formatted as `<key>=<value>`. No spaces. For every fractal the following fields are required (with example values):
```
formula=mandelbrot
res=300
xmin=-2.5
xmax=1.5
ymin=-1.5
ymax=1.5
```
### Zooming in
In order to zoom in on a region, click once in the top-left corner of the desired window, and again in the lower-right corner. The window will automatically resize and zoom so it displays the part of the fractal in that bounding box.

### Other parameters
There is an optional parameter `cmap` that sets the colormap used. Right now the following colormaps are supported (not case sensitive):
- grayscale
- magma
- rainbow
- grape

If a value for `cmap` is not specified, the default is grayscale.