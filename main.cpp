#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif
#pragma once
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <string>
#include <sstream>
#include <map>
#include <variant>
#include <algorithm>

// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui/highgui.hpp>
#define IM_HEIGHT 1000
#define IM_WIDTH 1000
#define N_ITER 384
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

Eigen::ArrayXXf mandelbrot(const Eigen::ArrayXXcf& complex_coords){
  Eigen::ArrayXXcf z_arr(IM_HEIGHT, IM_WIDTH);  //the array of z values for each coordinate
  Eigen::ArrayXXi all_i (IM_HEIGHT, IM_WIDTH);
  Eigen::ArrayXXcf other(IM_HEIGHT, IM_WIDTH);  //temp array to hold output of iteration
  Eigen::ArrayXXi escapetime(IM_HEIGHT, IM_WIDTH); //array to store the number of iterations it took each coord to escape (|z|>2)

  for(int i=1;i<=N_ITER;i++){
    other = z_arr*z_arr + complex_coords;
    all_i = all_i+1;
    // std::cout << other << "\n";
    //escapetime = escaped on this iteration ? i : 0
    // std::cout << escapetime << "\n";
    escapetime = ((z_arr.abs() >= 2)&&(escapetime == 0)).select(all_i, escapetime);
    z_arr = other;
  }
  return escapetime.cast<float>();
}

Eigen::ArrayXXf mandelbrot_cv(){
  Eigen::VectorXf xs = Eigen::VectorXf::LinSpaced(IM_WIDTH, -2, 2); //x coordinates in order (for meshgrid)
  Eigen::VectorXf ys = Eigen::VectorXf::LinSpaced(IM_HEIGHT, -2, 2.0);  //y coordinates in order (for meshgrid)
  Eigen::ArrayXXi escapetime(IM_HEIGHT, IM_WIDTH); //array to store the number of iterations it took each coord to escape (|z|>2)
  for(int i=0; i<IM_WIDTH; i++){
    for(int j=0; j<IM_HEIGHT; j++){
      float a=0, b=0;
      float atmp;
      int c=1;
      while(a*a+b*b<4 && c<=N_ITER){
        atmp = a*a - b*b + xs(i);
        b = (a+a)*b + ys(j);
        a = atmp;
        c++;
      }
      escapetime(j,i) = c>N_ITER ? 0 : c;
    }
  }
  return escapetime.cast<float>();
}

Eigen::ArrayXXf julia(const double& c_real, 
                      const double& c_imag,
                      const double& xmin,
                      const double& xmax,
                      const double& ymin,
                      const double& ymax,
                      const double& res){
  int width = (int)((xmax-xmin)*res);
  int height = (int)((ymax-ymin)*res);
  Eigen::VectorXd xs = Eigen::VectorXd::LinSpaced(width, xmin, xmax); //x coordinates in order (for meshgrid)
  Eigen::VectorXd ys = Eigen::VectorXd::LinSpaced(height, ymin, ymax);  //y coordinates in order (for meshgrid)
  Eigen::ArrayXXi escapetime(height, width); //array to store the number of iterations it took each coord to escape (|z|>2)
  for(int i=0; i<width; i++){
    for(int j=0; j<height; j++){
      float a=xs(i);
      float b=ys(j);
      float atmp;
      int n=1;
      while(a*a+b*b<4 && n<=N_ITER){
        atmp = a*a - b*b + c_real;
        b = (a+a)*b + c_imag;
        a = atmp;
        n++;
      }
      escapetime(j,i) = n>N_ITER ? 0 : n;
    }
  }
  Eigen::ArrayXXf et_f = escapetime.cast<float>();
  return et_f/N_ITER;
}

// TODO make this use a common class so there isn't repeated code
Eigen::ArrayXXf burning_ship(const double& xmin,
                            const double& xmax,
                            const double& ymin,
                            const double& ymax,
                            const double& res){
  int width = (int)((xmax-xmin)*res);
  int height = (int)((ymax-ymin)*res);
  Eigen::VectorXd xs = Eigen::VectorXd::LinSpaced(width, xmin, xmax); //x coordinates in order (for meshgrid)
  Eigen::VectorXd ys = Eigen::VectorXd::LinSpaced(height, ymin, ymax);  //y coordinates in order (for meshgrid)
  Eigen::ArrayXXi escapetime(height, width); //array to store the number of iterations it took each coord to escape (|z|>2)
  for(int i=0; i<width; i++){
    for(int j=0; j<height; j++){
      float a=0;
      float b=0;
      float atmp;
      int n=1;
      while(a*a+b*b<4 && n<=N_ITER){
        a = abs(a);
        b = abs(b);
        atmp = a*a - b*b + xs(i);
        b = (a+a)*b + ys(j);
        a = atmp;
        n++;
      }
      escapetime(j,i) = n>N_ITER ? 0 : n;
    }
  }
  Eigen::ArrayXXf et_f = escapetime.cast<float>();
  return et_f/N_ITER;
}

Eigen::ArrayXXf julia(const double& c_real, const double& c_imag){
  double xmin=-2, xmax=2;
  double ymin=-1, ymax=1;
  return julia(c_real, c_imag, xmin, xmax, ymin, ymax, 300.0);
}

Eigen::ArrayXXf julia(const double& c_real, const double& c_imag, const double& res){
  double xmin=-2, xmax=2;
  double ymin=-1, ymax=1;
  return julia(c_real, c_imag, xmin, xmax, ymin, ymax, res);
}

std::string toLower(std::string s){
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  return s;
}

const int MAGMA = 0;
const double magma [5] = {0x003f5c,0x58508d,0xbc5090,0xff6361,0xffa600};
const int magma_len = 5;

double interp_color(const int cmap, const double val, const double min, const double max){
  const double* cmap_vals;
  double nvals;
  if(val<min || val>max){
    throw std::invalid_argument("val must be between min and max");
  }
  double sval = (val - min)/(max - min);
  if (cmap==MAGMA){
    cmap_vals = magma;
    nvals = (double)magma_len;
  }

  double lower, upper, subscaled;
  for(int i=0; i<nvals-1; i++){
    lower = ((double)i)/nvals;
    upper = ((double)(i+1))/nvals;
    if(sval>=lower && sval<upper){
      subscaled = (sval-lower)/(upper-lower);
      return (1-subscaled)*(*(cmap_vals+i+1)) + subscaled*(*(cmap_vals+i));
    }
  }
  return *(cmap_vals + (((int)nvals)-1));
}

class Fractal {
  public:
    int formula;
    float nf;
    double xmin, xmax, ymin, ymax, res;
    int width, height;
    Eigen::VectorXd xs, ys;
    Eigen::ArrayXXd escapetime;
    Fractal(int f, double xm, double xx, double ym, double yx, double r);
    Fractal(const std::string& s);
    Eigen::ArrayXXd compute(const int& n_iter);
  private:
    void otherInit();

};

void Fractal::otherInit(){
  width = (int)((xmax-xmin)*res);
  height = (int)((ymax-ymin)*res);
  xs = Eigen::VectorXd::LinSpaced(width, xmin, xmax); //x coordinates in order (for meshgrid)
  ys = Eigen::VectorXd::LinSpaced(height, ymin, ymax).reverse();  //y coordinates in order (for meshgrid)
}

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
    } else {
      *value=stof(config.at(name));
    }
}

// create fractal from config file; argument is filename
Fractal::Fractal(const std::string& conffile){
  std::map<std::string, std::string> config;
  std::ifstream cFile("fractalconf.cfg");

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

  readFromConf(config, "res", &res);
  readFromConf(config, "xmin", &xmin);
  readFromConf(config, "xmax", &xmax);
  readFromConf(config, "ymin", &ymin);
  readFromConf(config, "ymax", &ymax);
  otherInit();
}

Eigen::ArrayXXd Fractal::compute(const int& n_iter){
  escapetime = Eigen::ArrayXXd(height, width);
  float a=0, b=0;
  float atmp;
  for(int i=0; i<width; i++){
    for(int j=0; j<height; j++){
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
      } // else Julia, but that needs additional arguments (c_real and c_imag)
      if(formula==FRACTAL_MANDELBROT){
        nf = (float)n;
        if (n>n_iter){
          escapetime(j,i) = 0;
        } else {
          nf = nf - log2(log2(a*a+b*b)/2)/ log2(2.0f);
          escapetime(j,i) = nf;
        }
      } else {
        escapetime(j,i) = n>n_iter ? 0 : n;
      }
    }
  }
  return escapetime/((float)n_iter);
}

class JuliaSet : public Fractal {
  public:
    JuliaSet(double xm, double xx, double ym, double yx, double r):Fractal(FRACTAL_JULIA,xm,xx,ym,yx,r){};
    Eigen::ArrayXXd compute(const int& n_iter, const double& c_real, const double& c_imag);
};

Eigen::ArrayXXd JuliaSet::compute(const int& n_iter, const double& c_real, const double& c_imag){
  Eigen::ArrayXXi escapetime(height, width); //array to store the number of iterations it took each coord to escape (|z|>2)
  for(int i=0; i<width; i++){
    for(int j=0; j<height; j++){
      float a=xs(i);
      float b=ys(j);
      float atmp;
      int n=1;
      while(a*a+b*b<4 && n<=n_iter){
        atmp = a*a - b*b + c_real;
        b = (a+a)*b + c_imag;
        a = atmp;
        n++;
      }
      escapetime(j,i) = n>n_iter ? 0 : n;
    }
  }
  Eigen::ArrayXXd et_f = escapetime.cast<double>();
  return et_f/n_iter;
};

// cv::Mat colormap_fractal(Eigen::ArrayXXd& et_normed){
//   cv::Mat imageR, imageG, imageB;
//   et_normed
// }

int main()
{
  Eigen::setNbThreads(8);

  // JuliaFractal frac = JuliaFractal(-2,2,-2,2,300.0);
  // Fractal frac = Fractal(FRACTAL_BURNINGSHIP, -2,2,-2,2,300.0);
  Fractal frac=Fractal("fractalconf.cfg");
  // Eigen::ArrayXXd escapetime_normed=frac.compute(N_ITER, -0.512511498387847167, 0.521295573094847167);
  Eigen::ArrayXXd escapetime_normed=frac.compute(N_ITER);

  cv::namedWindow("Fractal", cv::WINDOW_NORMAL);
  cv::Mat image;

  Eigen::MatrixXd etn(escapetime_normed);

  double color = interp_color(MAGMA, 0.5, 0, 1);
  printf("Color: %f", color);

  etn = etn*(255);
  cv::eigen2cv(etn, image);
  image.convertTo(image, CV_8UC1);
  cv::Mat color_img(image);
  cv::applyColorMap(image, color_img, cv::COLORMAP_JET);
  cv::imshow("Fractal", color_img);
  cv::waitKey(0);
  return 0;
}