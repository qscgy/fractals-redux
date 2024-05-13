#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif
#include "colormap.h"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <map>
#include <variant>
#include <algorithm>
#include <cmath>

#define N_ITER 400
#define FRACTAL_MANDELBROT 1
#define FRACTAL_JULIA 2
#define FRACTAL_BURNINGSHIP 3
#define FRACTAL_MULTIJULIA 4

// using namespace std;

std::vector<double> linspace(double min, double max, int n)
{
	std::vector<double> result;
	int iterator = 0;

	for (int i = 0; i <= n-2; i++)	
	{
		double temp = min + i*(max-min)/(floor((double)n) - 1);
		result.insert(result.begin() + iterator, temp);
		iterator += 1;
	}
	result.insert(result.begin() + iterator, max);
	return result;
}

const double pi = std::acos(-1.0);

std::string toLower(std::string s){
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  return s;
}

class Fractal {
  public:
    int formula;
    std::string formulaStr, cmapStr;
    double xmin, xmax, ymin, ymax, res;
    int width, height;
    double homeParams[5];
    int topLeft[2], lowerRight[2];
    bool onSecondClick;
    Colormap cmap;
    double c_real, c_imag;
    double order_n;
    std::vector<double> xs, ys;
    cv::Mat escapetime;
    cv::Mat imageR;
    Fractal(int f, double xm, double xx, double ym, double yx, double r);
    Fractal(const std::string& s);
    void compute(const int& n_iter);
    void colorize(const double& val, const int& i, const int& j);
    void otherInit();
};

void Fractal::otherInit(){
  width = (int)((xmax-xmin)*res);
  height = (int)((ymax-ymin)*res);
  escapetime = cv::Mat(cv::Size(width, height), CV_64F);

  onSecondClick = false;
  xs = linspace(xmin, xmax, width);
  ys = linspace(ymin, ymax, height);
  std::reverse(ys.begin(), ys.end());
  imageR = cv::Mat(height, width, CV_8UC3, cv::Scalar(0,0,0));
}

/**
 * Constructs an object representing a fractal.
 * @param f formula code
 * @param xm xmin
 * @param xx xmax
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
  std::ifstream cFile(conffile);

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
    formulaStr = toLower(config.at("formula"));
    if(formulaStr=="mandelbrot"){
      formula=FRACTAL_MANDELBROT;
    } else if(formulaStr=="julia"){
      formula=FRACTAL_JULIA;
    } else if(formulaStr=="burningship"){
      formula=FRACTAL_BURNINGSHIP;
    } else if(formulaStr=="multijulia") {
      formula=FRACTAL_MULTIJULIA;
    } else {
      std::cerr << "The formula " << formulaStr << " is not a valid option.\n";
    }
  }

  if(config.find("cmap")==config.end()){
    cmapStr="grayscale";
  } else {
    cmapStr=config.at("cmap");
    if(palettes.find(cmapStr)==palettes.end()){
      cmapStr="grayscale";
    } // else, cmapStr is already set to a value that is in palettes
  }
  // cmap = {palettes.at(cmapStr).data(), (int)(palettes.at(cmapStr).size())};
  cmap = Colormap(cmapStr);

  // read in the rest of the fields
  readFromConf(config, "res", &res);
  readFromConf(config, "xmin", &xmin);
  readFromConf(config, "xmax", &xmax);
  readFromConf(config, "ymin", &ymin);
  readFromConf(config, "ymax", &ymax);
  homeParams[0] = xmin;
  homeParams[1] = xmax;
  homeParams[2] = ymin;
  homeParams[3] = ymax;
  homeParams[4] = res;

  if(((formula==FRACTAL_JULIA || formula==FRACTAL_MULTIJULIA) && config.find("c_real")!=config.end() && config.find("c_imag")!=config.end())){
    readFromConf(config, "c_real", &c_real);
    readFromConf(config, "c_imag", &c_imag);
  } else if(formula==FRACTAL_JULIA){
    std::cerr << "Must provide both c_real and c_imag for formula 'julia'." << std::endl;
    exit(1);
  } 
  if(formula==FRACTAL_MULTIJULIA){
    if(config.find("order") != config.end()){
      readFromConf(config, "order", &order_n);
    } else {
      order_n = 2;
    }
  }

  otherInit();
}

 void Fractal::compute(const int& n_iter){
  double a=0, b=0;
  double atmp, nf;
  long colorval;
  double* ptr;
  const double nOver2 = order_n / 2.0;
  for(int j=0; j<height; j++){
    ptr = escapetime.ptr<double>(j);
    for(int i=0; i<width; i++){
      int n=1;
      a=0;
      b=0;
      atmp=0;
      if(formula==FRACTAL_MANDELBROT){
        while(a*a+b*b<4 && n<=n_iter){
          atmp = a*a - b*b + xs.at(i);
          b = (a+a)*b + ys.at(j);
          a = atmp;
          n++;
        }
      } else if(formula==FRACTAL_BURNINGSHIP){
        a=xs.at(i);
        b=ys.at(j);
        while(a*a+b*b<4 && n<=n_iter){
          atmp = a*a - b*b - xs.at(i);
          b = abs(2.0*a*b) - ys.at(j);
          a = atmp;
          n++;
        }
      } else if(formula==FRACTAL_JULIA){
        a=xs.at(i);
        b=ys.at(j);
        while(a*a+b*b<4 && n<=n_iter){
          atmp = a*a - b*b + c_real;
          b = (a+a)*b + c_imag;
          a = atmp;
          n++;
        }
      } else if(formula==FRACTAL_MULTIJULIA){
        a=xs.at(i);
        b=ys.at(j);
        while(a*a+b*b<4 && n<=n_iter){
          atmp = pow((a * a + b * b), nOver2) * cos(order_n * atan2(b, a)) + c_real;
	        b = pow((a * a + b * b), nOver2) * sin(order_n * atan2(b, a)) + c_imag;
	        a = atmp;
          n++;
        }
      }
      if(formula==FRACTAL_MANDELBROT){
        nf = (double)n;
        if (n>=n_iter){
          ptr[i] = 1.0;
        } else {
          nf = nf - log(log(a*a+b*b)/2.0f)/log(2);
          ptr[i] = nf/((double)n_iter);
        }
      } else if(formula==FRACTAL_JULIA){
        nf = (double)n;
        if (n>=n_iter){
          ptr[i] = 1.0;
        } else {
          nf = nf + 1 - log(log(a*a+b*b))/log(2);
          ptr[i] = nf/((double)n_iter);
        }
      } else {
        ptr[i] = n>=n_iter ? 1.0 : n/((double)n_iter);
      }
      colorize(ptr[i], j, i);
    }
  }
  // return escapetime/((double)n_iter);
}

void Fractal::colorize(const double& val, const int& i, const int& j){
  long colorval_l = std::lround(interp_color(cmap, val));
  cv::Vec3b& pixel = imageR.at<cv::Vec3b>(i, j);
  for(int c=0; c<3;c++){
    pixel[c] = (colorval_l >> (c * 8)) & 0xFF;  // set BGR pixel values
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
        frac->xmin = frac->xs.at(frac->topLeft[0]);
        frac->xmax = frac->xs.at(frac->lowerRight[0]);
      } else {
        frac->xmin = frac->xs.at(frac->lowerRight[0]);
        frac->xmax = frac->xs.at(frac->topLeft[0]);
      }
      if(frac->lowerRight[1] > frac->topLeft[1]){
        frac->ymax = frac->ys.at(frac->topLeft[1]);
        frac->ymin = frac->ys.at(frac->lowerRight[1]);
      } else {
        frac->ymax = frac->ys.at(frac->lowerRight[1]);
        frac->ymin = frac->ys.at(frac->topLeft[1]);
      }
      frac->res = std::max(frac->width/(frac->xmax - frac->xmin), frac->height/(frac->ymax - frac->ymin));

      frac->otherInit();
      frac->compute(N_ITER);
      cv::imshow("Fractal", frac->imageR);
    } else {
      frac->topLeft[0] = x;
      frac->topLeft[1] = y;
      frac->onSecondClick = true;
    }
    // std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
  } else if(event==cv::EVENT_RBUTTONDOWN){
    frac->xmin=frac->homeParams[0];
    frac->xmax=frac->homeParams[1];
    frac->ymin=frac->homeParams[2];
    frac->ymax=frac->homeParams[3];
    frac->res=frac->homeParams[4];
    frac->otherInit();
    frac->compute(N_ITER);
    cv::imshow("Fractal", frac->imageR);
  }
}

int main(int argc, char **argv)
{
  if(argc<2){
    std::cerr << "Must pass config file path." << std::endl;
    exit(1);
  }
  
  Fractal frac = Fractal(argv[1]);
  frac.compute(N_ITER);
  
  cv::namedWindow("Fractal", cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback("Fractal", editWindowCallback, &frac);
  cv::imshow("Fractal", frac.imageR);
  cv::waitKey(0);

  if(argc >= 3){  // save if a file path was provided
    char* result;
    const char *fmt_str;
    if(frac.formula!=FRACTAL_JULIA){
    fmt_str = "%s/%s_%s_%.5g_%.5g_res%.7g.png";
    asprintf(&result, fmt_str, argv[2], frac.formulaStr.c_str(), 
        frac.cmapStr.c_str(), frac.xmin, frac.ymin, frac.res);
    } else {
      fmt_str = "%s/%s_%s_%.6g+%.6gi_%.5g_%.5g_res%.7g.png";
    asprintf(&result, fmt_str, argv[2], frac.formulaStr.c_str(), 
        frac.cmapStr.c_str(), frac.c_real, frac.c_imag, frac.xmin, frac.ymin, frac.res);
    }
    cv::imwrite(result, frac.imageR);
  }
  return 0;
}