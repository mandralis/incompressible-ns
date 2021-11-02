#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <eigen-io.h>

#define _USE_MATH_DEFINES

using Eigen::MatrixXd;
using Eigen::Vector2d;

class Simulation {
public:
    // Physics
    double Re;
    // Spatial discretization
    double Lx;
    double Ly;
    int Nx;
    int Ny;
    int Nxs;
    int Nys;
    double dx;
    double dy;
    // Temporal discretization
    double T;
    int Nt;
    double dt;
    // Velocity field and pressure field in row major order on staggered grid
    // Keep matrices dynamic for now...don't know how to deal with fixed size matrices in classes
    MatrixXd u;
    MatrixXd v;
    MatrixXd p;
    // Interpolated quantities
    MatrixXd uq;
    MatrixXd vq;
    // Derivatives
    MatrixXd lapU;
    MatrixXd lapV;
    MatrixXd duvdx;
    MatrixXd duvdy;
    MatrixXd duvdxC;
    MatrixXd duvdyC;
    MatrixXd duudx;
    MatrixXd dvvdy;

    // Constructor
    Simulation(double Re_, double Lx_, double Ly_, double T_, int Nx_, int Ny_, int Nt_)
      : Re(Re_), Lx(Lx_),Ly(Ly_),T(T_), Nx(Nx_),Ny(Ny_),Nt(Nt_),dx(Lx_/Nx_),dy(Ly/Ny),dt(T/Nt),Nxs(Nx_-1),Nys(Ny_-1),
        u((Nx_-1)*(Ny_-1),1), v((Nx_-1)*(Ny_-1),1), p((Nx_-1)*(Ny_-1),1),
        uq((Nx-1)*(Ny-1),1), vq((Nx-1)*(Ny-1),1),lapU((Nx-1)*(Ny-1),1), lapV((Nx-1)*(Ny-1),1),
        duvdx((Nx_-1)*(Ny_-1),1), duvdy((Nx_-1)*(Ny_-1),1), duvdxC((Nx_-1)*(Ny_-1),1), duvdyC((Nx_-1)*(Ny_-1),1),
        duudx((Nx_-1)*(Ny_-1),1), dvvdy((Nx_-1)*(Ny_-1),1){}

    void interpUVFaceToCorner() {
        // Function that linearly interpolates the grid velocities from the left and bottom face to
        // the lower left corner
        int Nq = Nxs * Nys;
        for (int i=0;i<Nq;i++) {
            int idxU = i+Nxs;
            if (idxU > Nq-1) {idxU = idxU-Nq;}
            int idxV = i-1;
            if (i%Nxs==0) {idxV = idxV+Nxs;}
            uq(i) = 0.5 * (u(i) + u(idxU));
            vq(i) = 0.5 * (v(i) + v(idxV));
        }
    }

    void interpCrossTermCornerToFace() {
        // Function that linearly interpolates the cross derivative from the lower left corner to the left
        // and bottom face
        // duvdy should be on the left face
        // duvdx should be on the bottom face
        int Nq = Nxs * Nys;
        for (int i=0;i<Nq;i++) {
            int idxU = i+Nxs;
            if (idxU > Nq-1) {idxU = idxU-Nq;}
            int idxV = i-1;
            if (i%Nxs==0) {idxV = idxV+Nxs;}
            uq(i) = 0.5 * (u(i) + u(idxU));
            vq(i) = 0.5 * (v(i) + v(idxV));
        }
    }

    void computeLaplacian() {
        // Function that computes Laplacian on grid faces
        int Nq = Nxs * Nys;
        for (int i=0;i<Nq;i++) {
            int iX_ = i-1;
            int iXX = i+1;
            if(i%Nxs){iX_=iX_+Nxs;}
            if((i+1)%Nxs) {iXX=iXX-Nxs;}
            int iY_ = i-Nxs;
            int iYY = i+Nxs;
            if(iY_<0) {iY_=iY_+Nq;}
            if(iYY>Nq-1) {iYY=iYY-Nq;}
            lapU(i) = 1/(dx*dx) * (u(iXX)-2*u(i)+u(iX_)) +  1/(dy*dy) * (u(iYY)-2*u(i)+u(iY_));
            lapV(i) = 1/(dx*dx) * (v(iXX)-2*v(i)+v(iX_)) +  1/(dy*dy) * (v(iYY)-2*v(i)+v(iY_));
        }
    }

    void computeSquareTerm() {
        // Function that duudx and dvvdy on grid faces
        int Nq = Nxs * Nys;
        for (int i=0;i<Nq;i++) {
            int iX_ = i-1;
            int iXX = i+1;
            if(i%Nxs){iX_=iX_+Nxs;}
            if((i+1)%Nxs) {iXX=iXX-Nxs;}
            int iY_ = i-Nxs;
            int iYY = i+Nxs;
            if(iY_<0) {iY_=iY_+Nq;}
            if(iYY>Nq-1) {iYY=iYY-Nq;}
            duudx(i) = 1/(2*dx) * (u(iXX)*u(iXX)-u(iX_)*u(iX_));
            dvvdy(i) = 1/(2*dy) * (v(iYY)*v(iYY)-v(iY_)*v(iY_));
        }
    }

    void computeCrossTermAtCorners() {
        // Function that computes duvdx and duvdy on grid corners
        int Nq = Nxs * Nys;
        for (int i=0;i<Nq;i++) {
            int iX_ = i-1;
            int iXX = i+1;
            if(i%Nxs){iX_=iX_+Nxs;}
            if((i+1)%Nxs) {iXX=iXX-Nxs;}
            int iY_ = i-Nxs;
            int iYY = i+Nxs;
            if(iY_<0) {iY_=iY_+Nq;}
            if(iYY>Nq-1) {iYY=iYY-Nq;}
            duvdxC(i) = 1/(2*dx) * (uq(iXX)*vq(iXX)-uq(iX_)*vq(iX_));
            duvdyC(i) = 1/(2*dy) * (uq(iYY)*vq(iYY)-uq(iY_)*vq(iY_));
        }
    }
};

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
//
//MatrixXd interpolateU() {
//    // Function that linearly interpolates the grid
//    // Matrix containing interpolated values at the corners of the grid (green points)
//    int Nq = Nx*Ny;
//    MatrixXd uq(Nq,1);
//    int Nu = Nx*(Ny-1);
//    // First interpolate all interior points and deal with boundary points later.
//    // Reminder: we are interpolating to bottom left corner.
//    for (int i=0;i<Nq;i++) {
//        // First Nx points on the upper boundary
//        if (i<Nx) {
//            //Periodic boundary conditions
//            uq(i,1) = 0.5*(u(i,1) + u(Nu-Nx+i));
//        }
//        // Last Nx points on the lower boundary
//        if (i>=Nq-Nx) {
//            uq(i,1) = 0.5*(u(Nu-Nx+i,1) + u(i-Nu,1));
//        }
//        // For all other points
//        uq(i,1) = 0.5 * (u(i-Nx,1) + u(i,1));
//    }
//    return uq;
//}
//
//MatrixXd interpolateV() {
//    // Function that linearly interpolates the grid
//    // Matrix containing interpolated values at the corners of the grid (green points)
//    int Nq = Nx*Ny;
//    MatrixXd vq(Nq,1);
//    int Nv = (Nx-1)*Ny;
//    // First interpolate all interior points and deal with boundary points later.
//    // Reminder: we are interpolating to bottom left corner.
//    for (int i=0;i<Nq;i++) {
//        // First Nx points on the upper boundary
//        if (i<Nx) {
//            //Periodic boundary conditions
//            vq(i,1) = 0.5*(u(i,1) + u(Nv-Nx+i));
//        }
//        // Last Nx points on the lower boundary
//        if (i>=Nq-Nx) {
//            vq(i,1) = 0.5*(u(Nv-Nx+i,1) + u(i-Nv,1));
//        }
//        // For all other points
//        vq(i,1) = 0.5 * (u(i-Nx,1) + u(i,1));
//    }
//    return vq;
//}