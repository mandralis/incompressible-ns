#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <eigen-io.h>

#define _USE_MATH_DEFINES

using Eigen::MatrixXd;
using Eigen::Vector2d;

// Spring mass system RHS function
Vector2d compute_rhs_springmass(const Vector2d &u) {
    double m = 1.0; // careful...hardcoded
    double k = 1.0;
    Vector2d dudt(0.0,0.0);
    dudt(0) = u(1);
    dudt(1) = -k/m * u(0);
    return dudt;
};

Vector2d sol_springmass(const double t) {
    Vector2d u(0.0,0.0);
    u(0) = sqrt(2) * cos(t - M_PI/4.0);
    u(1) = -sqrt(2) * sin(t - M_PI/4.0);
    return u;
}

// Van der pol oscillator RHS function
Vector2d compute_rhs_van(const Vector2d &u) {
    double mu = 1.0; // careful...hardcoded
    Vector2d dudt(0.0,0.0);
    dudt(0) = u(0);
    dudt(1) = (mu*(1.0-u(0)*u(0))*u(1)-u(0));
    return dudt;
};

// Time advancing function
Vector2d time_advance_RK3(const Vector2d &u0, const double &dt) {
    Vector2d u(0.0,0.0);
    // Step 1
    Vector2d F = compute_rhs_springmass(u0);
    Vector2d k1 = dt * F;
    // Step 2
    F = compute_rhs_springmass(u0 + 0.5*k1);
    Vector2d k2 = dt * F;
    // Step 3
    F = compute_rhs_springmass(u0 - k1 + 2*k2);
    Vector2d k3 = dt * F;
    // Time advancement
    u = u0 + 1.0/6.0 * k1 + 2.0/3.0 * k2 + 1.0/6.0 * k3;
    return u;
}

// Time advancing function
Vector2d time_advance_euler(const Vector2d &u0, const double &dt) {
    Vector2d u(0.0,0.0);
    // Time advancement
    u = u0 + dt * compute_rhs_springmass(u0);
    return u;
}

int main() {
    double T = 15.0;
    // Test order of convergence
    int Nconv = 100;
    // Error matrix
    MatrixXd e_mat = MatrixXd::Zero(1,Nconv);
    // N matrix
    MatrixXd N_mat = MatrixXd::Zero(1,Nconv);

    for (int i=1;i<Nconv;i++) {
        std::cout << "N: " << i << endl;
        int Nt = 100*i;
        double dt = T/Nt;
        Vector2d u0(1.0,1.0);

        MatrixXd u_sol = MatrixXd::Zero(2,Nt+1);
        MatrixXd u_hat = MatrixXd::Zero(2,Nt+1);
        MatrixXd t_vec = MatrixXd::Zero(1,Nt+1);
        u_hat.block<2,1>(0,0) = u0;
        u_sol.block<2,1>(0,0) = u0;

        // Initialize the time
        double t = 0.0 + dt;
        // Initialize the error
        double e = 0.0;
        for (int j=1;j<=Nt;j++) {
            // Advance the simulation
            Vector2d u = time_advance_RK3(u0,dt);
            Vector2d u_exact = sol_springmass(t);
            // Write to approximate solution matrix
            u_hat.block<2,1>(0,j) = u;
            // Write to exact solution matrix
            u_sol.block<2,1>(0,j) = u_exact;
            // Write to time matrix
            t_vec(0,j) = t;
            // Update current velocity
            u0 = u;
            // Update the current time
            t += dt;
            // Compute the error
            e += pow(u(0,0)-u_exact(0,0),2.0) + pow(u(1,0)-u_exact(1,0),2.0);
        }
        // Store the values of the error
        e_mat(0,i) = sqrt(e/Nt);
        // Store the values of N
        N_mat(0,i) = Nt;
        saveData("u_hat.csv",u_hat);
        saveData("u_sol.csv",u_sol);
        saveData("t.csv",t_vec);
    }
    saveData("e.csv",e_mat);
    saveData("N.csv",N_mat);
    return 0;
}

//#include <nr3.h>
//#include <stepper.h>
//#include <odeint.h>
//#include <stepperdopr5.h>

//// RHS function (let's use a functor instead of a bare function)
//struct rhs_van {
//    double mu;
//    rhs_van(double mu_) : mu(mu_) {}
//    void operator() (const double t, const Vector2d &u, Vector2d &dudt) {
//        dudt[0] = u[1];
//        dudt[1] = (mu*(1.0-u[0]*u[0])*u[1]-u[0]);
//    }
//};

//int main() {
//    const int nvar=2;
//    const double atol=1.0e-3, rtol=atol, h1=0.01, hmin=0.0, x1=0.0, x2=2.0;
//    Vector2d ystart(nvar);
//    ystart[0]=2.0;
//    ystart[1]=0.0;
//    Output out(20);
//    rhs_van d(1.0e-3);
//    Odeint<StepperDopr5<rhs_van> > ode(ystart,x1,x2,atol,rtol,h1,hmin,out,d);
//    ode.integrate();
//
//    // Print the output
//    for (int i=0;i<out.count;i++)
//        cout << out.xsave[i] << " " << out.ysave[0][i] << " " << out.ysave[1][i] << endl;
//
//    return 0;
//}