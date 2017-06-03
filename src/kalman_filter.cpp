#include "kalman_filter.h"

#include <iostream>
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
    /**
        * predict the state
    */
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    /**
        * update the state by using Kalman Filter equations
    */
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;

    update_common_helper(y);
}


inline void normalize_angle(double& phi) {
    /*
     * adjust phi between -pi and pi;
     */
    phi = atan2(sin(phi), cos(phi));
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    /**
        * update the state by using Extended Kalman Filter equations
    */
    const double eps = 0.0001;

    //recover state parameters
    const double px = x_(0);
    const double py = x_(1);
    const double vx = x_(2);
    const double vy = x_(3);
    //pre-compute a set of terms to avoid repeated calculation
    const double c1 = px*px+py*py;
    const double c2 = std::max(eps, sqrt(c1));

    VectorXd hx = VectorXd(3);
    hx << c2, atan2(py,px), (px*vx+py*vy)/c2;

    VectorXd y = z - hx;
    normalize_angle(y(1));

    update_common_helper(y);
}


void KalmanFilter::update_common_helper(const VectorXd &y) {
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    //new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

