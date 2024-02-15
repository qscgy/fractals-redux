#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui/highgui.hpp>
#define IM_HEIGHT 1000
#define IM_WIDTH 1000
#define N_ITER 256

// using Eigen::ArrayXXd;
// using namespace Eigen;

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

Eigen::ArrayXXf julia(const double& c_real, const double& c_imag){
  double xmin=-2, xmax=2;
  double ymin=-2, ymax=2;
  Eigen::VectorXd xs = Eigen::VectorXd::LinSpaced(IM_WIDTH, xmin, xmax); //x coordinates in order (for meshgrid)
  Eigen::VectorXd ys = Eigen::VectorXd::LinSpaced(IM_HEIGHT, xmin, xmax);  //y coordinates in order (for meshgrid)
  Eigen::ArrayXXi escapetime(IM_HEIGHT, IM_WIDTH); //array to store the number of iterations it took each coord to escape (|z|>2)
  for(int i=0; i<IM_WIDTH; i++){
    for(int j=0; j<IM_HEIGHT; j++){
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
  return escapetime.cast<float>();
}

int main()
{
  Eigen::setNbThreads(8);
  // Eigen::ArrayXXf escapetime_normed = mandelbrot_cv()/N_ITER;
  Eigen::ArrayXXf escapetime_normed = julia(-0.4, 0.6)/N_ITER;
  cv::Mat image;
  Eigen::MatrixXf etn(escapetime_normed);
  cv::eigen2cv(etn, image);
  cv::imshow("Mandelbrot", image);
  cv::waitKey(0);
  return 0;
}