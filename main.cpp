#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif
#include "colormap.h"
#include <fstream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <string>
#include <sstream>
#include <map>
#include <variant>
#include <algorithm>
#include <cmath>

// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui/highgui.hpp>
#define IM_HEIGHT 1000
#define IM_WIDTH 1000
#define N_ITER 380
#define FRACTAL_JULIA 2
#define FRACTAL_MANDELBROT 1
#define FRACTAL_BURNINGSHIP 3

// using Eigen::ArrayXXd;
// using namespace Eigen;
// using namespace std;

// https://gitlab.math.ethz.ch/NumCSE/NumCSE/blob/3c723d06ffacab3dc45726ad0a95e33987dc35aa/Utils/meshgrid.hpp
//! Generates a mesh, just like Matlab's meshgrid
//  Template specialization for column vectors (Eigen::VectorXd)
//  in : x, y column vectors 
//       X, Y matrices, used to save the mesh
template <typename Scalar>
void meshgrid(const Eigen::Matrix<Scalar, -1, 1>& x, 
              const Eigen::Matrix<Scalar, -1, 1>& y,
              Eigen::Matrix<Scalar, -1, -1>& X,
              Eigen::Matrix<Scalar, -1, -1>& Y) {
  const long nx = x.size(), ny = y.size();
  X.resize(ny, nx);
  Y.resize(ny, nx);
  for (long i = 0; i < ny; ++i) {
    X.row(i) = x.transpose();
  }
  for (long j = 0; j < nx; ++j) {
    Y.col(j) = y;
  }
}

const double pi = std::acos(-1.0);

//! Generates a mesh, just like Matlab's meshgrid
//  Template specialization for row vectors (Eigen::RowVectorXd)
//  in : x, y row vectors 
//       X, Y matrices, used to save the mesh
template <typename Scalar>
void meshgrid(const Eigen::Matrix<Scalar, 1, -1>& x, 
              const Eigen::Matrix<Scalar, 1, -1>& y,
              Eigen::Matrix<Scalar, -1, -1>& X,
              Eigen::Matrix<Scalar, -1, -1>& Y) {
  Eigen::Matrix<Scalar, -1, 1> xt = x.transpose(),
                               yt = y.transpose();
  meshgrid(xt, yt, X, Y);
}

std::string toLower(std::string s){
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  return s;
}

class Fractal {
  public:
    int formula;
    float nf;
    double xmin, xmax, ymin, ymax, res;
    int width, height;
    int topLeft[2], lowerRight[2];
    bool onSecondClick;
    int cmap;
    double c_real, c_imag;
    Eigen::VectorXd xs, ys;
    Eigen::ArrayXXd escapetime;
    cv::Mat imageR;
    Fractal(int f, double xm, double xx, double ym, double yx, double r);
    Fractal(const std::string& s);
    Eigen::ArrayXXd compute(const int& n_iter);
    void colorize(const double& val, const int& i, const int& j);
    void otherInit();
};

void Fractal::otherInit(){
  width = (int)((xmax-xmin)*res);
  height = (int)((ymax-ymin)*res);
  onSecondClick = false;
  xs = Eigen::VectorXd::LinSpaced(width, xmin, xmax); //x coordinates in order (for meshgrid)
  ys = Eigen::VectorXd::LinSpaced(height, ymin, ymax).reverse();  //y coordinates in order (for meshgrid)
  imageR = cv::Mat(height, width, CV_8UC3, cv::Scalar(0,0,0));
}

/*
 * Constructs an object representing a fractal.
*/
Fractal::Fractal(int f, double xm, double xx, double ym, double yx, double r){
  formula=f;
  xmin=xm;
  xmax=xx;
  ymin=ym;
  ymax=yx;
  res=r;
  otherInit();
}

void readFromConf(const std::map<std::string, std::string>& config, const std::string& name, double* value){
    if(config.find(name)==config.end()){
      std::cerr << name << " misisng from config.";
      exit(1);
    } else {
      // std::cout << name << "\n";
      *value=stof(config.at(name));
    }
}

// create fractal from config file; argument is filename
Fractal::Fractal(const std::string& conffile){
  std::map<std::string, std::string> config;
  std::ifstream cFile("fractalconf.cfg");

  // read in fields from config file
  if (cFile.is_open())
  {
    std::string line;
    while(getline(cFile, line)){
      line.erase(std::remove_if(line.begin(), line.end(), isspace),
                            line.end());
      if(line[0] == '#' || line.empty())
        continue;
      auto delimiterPos = line.find("=");
      std::string name = line.substr(0, delimiterPos);
      std::string valueStr = line.substr(delimiterPos + 1);
      // std::cout << name << " " << valueStr << '\n';
      config.insert({name, valueStr});
    }
  }
  else {
    std::cerr << "Couldn't open config file for reading.\n";
  }

  // read string-valued fields first
  if(config.find("formula")==config.end()){
    std::cerr << "formula missing from config.\n";
  } else {
    const std::string formulaStr = config.at("formula");
    if(toLower(formulaStr)=="mandelbrot"){
      formula=FRACTAL_MANDELBROT;
    } else if(toLower(formulaStr)=="julia"){
      formula=FRACTAL_JULIA;
    } else if(toLower(formulaStr)=="burningship"){
      formula=FRACTAL_BURNINGSHIP;
    } else {
      std::cerr << "The formula " << formulaStr << " is not a valid option.\n";
    }
  }
  if(config.find("cmap")==config.end()){
    cmap=GRAYSCALE;
  } else {  // select the colormap
    const std::string cmapStr = toLower(config.at("cmap"));
    if(cmapStr=="magma"){
      cmap=MAGMA;
    } else if(cmapStr=="rainbow"){
      cmap=RAINBOW;
    } else if(cmapStr=="grape"){
      cmap=GRAPE;
    } else {
      cmap=GRAYSCALE;
    }
  }

  // read in the rest of the fields
  readFromConf(config, "res", &res);
  readFromConf(config, "xmin", &xmin);
  readFromConf(config, "xmax", &xmax);
  readFromConf(config, "ymin", &ymin);
  readFromConf(config, "ymax", &ymax);

  if((formula==FRACTAL_JULIA && config.find("c_real")!=config.end() && config.find("c_imag")!=config.end())){
    readFromConf(config, "c_real", &c_real);
    readFromConf(config, "c_imag", &c_imag);
  } else if(formula==FRACTAL_JULIA){
    std::cerr << "Must provide both c_real and c_imag for formula 'julia'." << std::endl;
    exit(1);
  }

  otherInit();
}

Eigen::ArrayXXd Fractal::compute(const int& n_iter){
  escapetime = Eigen::ArrayXXd(height, width);
  float a=0, b=0;
  float atmp;
  long colorval;
  // printf("width=%d  height=%d\n", width, height);
  for(int j=0; j<height; j++){
    for(int i=0; i<width; i++){
      int n=1;
      a=0;
      b=0;
      atmp=0;
      if(formula==FRACTAL_MANDELBROT){
        while(a*a+b*b<4 && n<=n_iter){
          atmp = a*a - b*b + xs(i);
          b = (a+a)*b + ys(j);
          a = atmp;
          n++;
        }
      } else if(formula==FRACTAL_BURNINGSHIP){
        a=xs(i);
        b=ys(j);
        while(a*a+b*b<4 && n<=n_iter){
          atmp = a*a - b*b - xs(i);
          b = abs(2.0*a*b) - ys(j);
          a = atmp;
          n++;
        }
      } else if(formula==FRACTAL_JULIA){
        a=xs(i);
        b=ys(j);
        while(a*a+b*b<4 && n<=n_iter){
          atmp = a*a - b*b + c_real;
          b = (a+a)*b + c_imag;
          a = atmp;
          n++;
        }
      }
      if(formula==FRACTAL_MANDELBROT){
        nf = (float)n;
        if (n>=n_iter){
          escapetime(j,i) = 1.0;
        } else {
          // printf("%f\n", log(log(a*a+b*b)/2.0f)/log(2));
          nf = nf - log(log(a*a+b*b)/2.0f)/log(2);
          escapetime(j,i) = nf/((double)n_iter);
        }
      } else {
        escapetime(j,i) = n>=n_iter ? 1.0 : n/((double)n_iter);
      }
      colorize(escapetime(j,i), j, i);
    }
  }
  // return escapetime/((double)n_iter);
  return escapetime;
}

void Fractal::colorize(const double& val, const int& i, const int& j){
  // printf("hex color: #%x\n", val);
  long colorval_l = std::lround(interp_color(cmap, val));
  cv::Vec3b& pixel = imageR.at<cv::Vec3b>(i, j);
  for(int c=0; c<3;c++){
    pixel[c] = (colorval_l >> (c * 8)) & 0xFF;
    // printf("pixel[%d]=%d\n", i, pixel[i]);
  }
}

void editWindowCallback(int event, int x, int y, int flags, void* userdata){
  Fractal* frac = (Fractal*) userdata;
  if(event == cv::EVENT_LBUTTONDOWN){
    // x=right, y=down
    if(frac->onSecondClick){
      frac->lowerRight[0] = x;
      frac->lowerRight[1] = y;

      if(frac->lowerRight[0] > frac->topLeft[0]){
        frac->xmin = frac->xs(frac->topLeft[0]);
        frac->xmax = frac->xs(frac->lowerRight[0]);
      } else {
        frac->xmin = frac->xs(frac->lowerRight[0]);
        frac->xmax = frac->xs(frac->topLeft[0]);
      }
      if(frac->lowerRight[1] > frac->topLeft[1]){
        frac->ymax = frac->ys(frac->topLeft[1]);
        frac->ymin = frac->ys(frac->lowerRight[1]);
      } else {
        frac->ymax = frac->ys(frac->lowerRight[1]);
        frac->ymin = frac->ys(frac->topLeft[1]);
      }
      frac->res = frac->width/(frac->xmax - frac->xmin);

      frac->otherInit();
      frac->compute(N_ITER);
      cv::imshow("Fractal", frac->imageR);
    } else {
      frac->topLeft[0] = x;
      frac->topLeft[1] = y;
      frac->onSecondClick = true;
    }
    std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
  }
}

int main(int argc, char **argv)
{
  Fractal frac = Fractal("fractalconf.cfg");

  Eigen::setNbThreads(16);

  frac.compute(N_ITER);
  
  cv::namedWindow("Fractal", cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback("Fractal", editWindowCallback, &frac);
  cv::imshow("Fractal", frac.imageR);
  cv::waitKey(0);
  if(argc >= 2){  // save if a file path was provided
    const char *fmt_str = "%s/%.6f_%.6f_res%.1f.png";
    char* result;
    asprintf(&result, fmt_str, argv[1],frac.xmin,frac.ymin, frac.res);
    cv::imwrite(result, frac.imageR);
  }
  return 0;
}