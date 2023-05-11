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
#define IM_HEIGHT 601
#define IM_WIDTH 601
#define N_ITER 60

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

Eigen::ArrayXXi mandelbrot(const Eigen::ArrayXXcf& complex_coords){
  Eigen::ArrayXXcf z_arr(IM_HEIGHT, IM_WIDTH);  //the array of z values for each coordinate
  Eigen::ArrayXXi mask (IM_HEIGHT, IM_WIDTH); //all zeros
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
  return escapetime;
}

int main()
{
  Eigen::VectorXf xs = Eigen::VectorXf::LinSpaced(IM_WIDTH, -2.1, 2.1); //x coordinates in order (for meshgrid)
  Eigen::VectorXf ys = Eigen::VectorXf::LinSpaced(IM_HEIGHT, 2.1, -2.1);  //y coordinates in order (for meshgrid)
  Eigen::MatrixXf x_coords(IM_HEIGHT, IM_WIDTH), y_coords(IM_HEIGHT, IM_WIDTH);  //x and y coordinates of each pixel
  meshgrid(xs, ys, x_coords, y_coords);
  std::complex<float> I(0.0f, 1.0f);
  Eigen::ArrayXXcf complex_coords = x_coords.cast<std::complex<float> >()+ I*y_coords.cast<std::complex<float> >(); //coords defined as complex numbers
  Eigen::ArrayXXcf twos = Eigen::ArrayXXcf::Constant(IM_HEIGHT, IM_WIDTH, 2.5);
  
  // std::cout << escapetime;
  Eigen::ArrayXXi escapetime = mandelbrot(complex_coords);
  Eigen::ArrayXXf escapetime_normed = escapetime.cast<float>()/N_ITER;
  cv::Mat image;
  Eigen::MatrixXf etn(escapetime_normed);
  cv::eigen2cv(etn, image);
  cv::imshow("Mandelbrot", image);
  cv::waitKey(0);
  return 0;
}