#include <iostream>
#include <Eigen/Dense>
#define IM_HEIGHT 401
#define IM_WIDTH 401
#define N_ITER 60

// using Eigen::ArrayXXd;

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

int main()
{
  ArrayXXcf z_arr(IM_HEIGHT, IM_WIDTH);  //the array of z values for each coordinate
  MatrixXi escapetime(IM_HEIGHT, IM_WIDTH); //array to store the number of iterations it took each coord to escape (|z|>2)
  VectorXd xs = VectorXd::LinSpaced(IM_WIDTH, -2.1, 2.1);
  VectorXd ys = VectorXd::LinSpaced(IM_HEIGHT, -2.1, 2.1);
  MatrixXd x_coords(IM_HEIGHT, IM_WIDTH), y_coords(IM_HEIGHT, IM_WIDTH);  //x and y coordinates of each pixel
  meshgrid(xs, ys, x_coords, y_coords);
  ArrayXXcf complex_coords = x_coords.cast<std::complex<float>>() + 1.0if*y_coords.cast<std::complex<float>>(); //coords defined as complex numbers
  ArrayXXcf other = (IM_HEIGHT, IM_WIDTH);  //temp array to hold output of iteration
  
  for(int i=0;i<N_ITER;i++){
    other = z_arr*z_arr + complex_coords;
    z_arr = z_arr.min(other);
  }
}