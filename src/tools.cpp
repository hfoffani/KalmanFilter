#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    /**
        * Calculate the RMSE here.
    */

    VectorXd rmse(4);
    rmse << 0,0,0,0;

    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    if (estimations.size() == 0) {
        cout << "Error: estimation vector has zero size." << endl;
        return rmse;
    }
    if (ground_truth.size() != estimations.size()) {
        cout << "Error: vectors have different sizes" << endl;
        return rmse;
    }

    //accumulate squared residuals
    for(int i=0; i < estimations.size(); ++i){
        VectorXd diff = estimations[i] - ground_truth[i];
        rmse = rmse.array() + diff.array() * diff.array();
    }

    //calculate the mean
    rmse = rmse.array() / estimations.size();

    //calculate the squared root
    rmse = rmse.array().sqrt();

    //return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    /**
        * Calculate a Jacobian here.
    */
    const float eps = 0.0001;
    MatrixXd Hj(3,4);

    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    //pre-compute a set of terms to avoid repeated calculation
    float c1 = px*px+py*py;
    float c2 = sqrt(c1);
    float c3 = (c1*c2);

    //check division by zero
    if (fabs(c1) < eps) {
        cout << "CalculateJacobian () - Error - Division by Zero" << endl;
        return Hj;
    }

    //compute the Jacobian matrix
    Hj << (px/c2), (py/c2), 0, 0,
          -(py/c1), (px/c1), 0, 0,
          py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

    return Hj;
}
