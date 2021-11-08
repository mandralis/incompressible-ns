//
// Created by Yanni Mandralis on 11/6/21.
//

#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <fftw3.h>
#include <eigen-io.h>

#define _USE_MATH_DEFINES

using Eigen::MatrixXd;
using Eigen::Vector2d;
using namespace std;

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

class Grid {
public:
    double Lx;
    double Ly;
    int Nx;
    int Ny;
    int Nq;
    Eigen::MatrixXd gridX;
    Eigen::MatrixXd gridY;
    Eigen::MatrixXd kX;
    Eigen::MatrixXd kY;
    Eigen::MatrixXd kXmod;
    Eigen::MatrixXd kYmod;
    Eigen::MatrixXd phiExact;
    double error;
    double mean_phi;
    double mean_phi_exact;
    // FFTW setup
    fftw_complex *Q,*Q_hat,*phi,*phi_hat;
    fftw_plan forward_p;
    fftw_plan inverse_p;

    Grid(double Lx_, double Ly_, int Nx_, int Ny_) :
    Lx(Lx_),Ly(Ly_),Nx(Nx_),Ny(Ny_), Nq(Nx_*Ny_),gridX(Nx*Ny,1),gridY(Nx*Ny,1),
    phiExact(Nx*Ny,1), kX(Nx*Ny,1), kY(Nx*Ny,1), kXmod(Nx*Ny,1), kYmod(Nx*Ny,1), error(0.0), mean_phi(0.0),
    mean_phi_exact(0.0){
        // Don't do in place transforms...requires extra considerations
        Q = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
        Q_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
        phi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
        phi_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
        forward_p = fftw_plan_dft_2d(Nx, Ny, Q, Q_hat, FFTW_FORWARD, FFTW_ESTIMATE);
        inverse_p = fftw_plan_dft_2d(Nx, Ny, phi_hat, phi, FFTW_BACKWARD, FFTW_ESTIMATE);
        // Compute the grid coordinates in the constructor
        computeGridCoordinates();
    }

    void poissonSolver() {
        // Compute exact phi for comparison
        computeSol();
        // Compute RHS
        computeRHS();
        // Compute forward fourier transform of Q
        fftw_execute(forward_p);
        for (int i=1;i<Nq;i++){
            phi_hat[i][0] = -Q_hat[i][0] / (kXmod(i)*kXmod(i) + kYmod(i)*kYmod(i));
            phi_hat[i][1] = -Q_hat[i][1] / (kXmod(i)*kXmod(i) + kYmod(i)*kYmod(i));
        }
        // Set the fourier coefficient at (0,0) to an arbitrary value...it is just the mean
        phi_hat[0][0] = 1.0;
        phi_hat[0][1] = 1.0;
        // Compute the inverse fourier transform of phi_hat
        fftw_execute(inverse_p);
        // Normalize the elements of phi by 1/(Nx*Ny)
        // and compute the mean value
        mean_phi = 0.0;
        for (int i=0;i<Nq;i++) {
            phi[i][0] = phi[i][0]/(Nx*Ny);
            phi[i][1] = phi[i][1]/(Nx*Ny);
            mean_phi += phi[i][0];
        }
        mean_phi /= Nq;
        // Compute the error
        error = 0.0;
        for (int i=0;i<Nq;i++) {
            error += abs((phi[i][0]-mean_phi)-(phiExact(i)-mean_phi_exact));
        }
        error = error/Nq;
        fftw_destroy_plan(forward_p);
        fftw_destroy_plan(inverse_p);
        fftw_cleanup();
    }

    void computeGridCoordinates() {
        // Grid goes from -Lx/2 <= x <= Lx/2, -Ly/2 <= y <= Ly/2
        double dx = Lx/Nx;
        double dy = Ly/Ny;
        double y = Ly/2.0;
        for (int j=0;j<Ny;j++) {
            double x = -Lx/2.0;
            for (int i=0;i<Nx;i++) {
                gridX(i+Nx*j) = x;
                gridY(i+Nx*j) = y;
                kX(i+Nx*j) = 2*M_PI/Lx*i;
                kY(i+Nx*j) = 2*M_PI/Ly*j;
                if(i>Nx/2){kX(i+Nx*j)=-(Nx-i)*2*M_PI/Lx;}
                if(j>Ny/2){kY(i+Nx*j)=-(Ny-j)*2*M_PI/Ly;}
                x+=dx;
                // Compute the modified wavenumbers
                kXmod(i+Nx*j) = 2/dx*sin(kX(i+Nx*j)*dx/2)*sgn(kX(i+Nx*j));
                kYmod(i+Nx*j) = 2/dy*sin(kY(i+Nx*j)*dy/2)*sgn(kY(i+Nx*j));
            }
            y-=dy;
        }
    }

    void computeRHS() {
        // Everything in FFTW is in row major order
        for (int i=0;i<Nq;i++) {
            Q[i][0] = computeQ(gridX(i),gridY(i));
            Q[i][1] = 0.0; // complex part is zero
        }
    }

    void computeSol() {
        // Everything in FFTW is in row major order
        mean_phi_exact = 0.0;
        for (int i=0;i<Nq;i++) {
            phiExact(i) = computePhiExact(gridX(i),gridY(i));
            mean_phi_exact += phiExact(i);
        }
        mean_phi_exact /= Nq;
    }

    static double computeQ (double x,double y) {
        return M_PI*M_PI*exp(cos(M_PI*x)+cos(M_PI*y)) * (-sin(M_PI*x)*sin(M_PI*x)+cos(M_PI*x)-sin(M_PI*y)*sin(M_PI*y)+cos(M_PI*y));
    }

    static double computePhiExact(double x, double y) {
        return -exp(cos(M_PI*x)+cos(M_PI*y));
    }

    void writeToCSV() {
        // Writes the solution phi to a CSV file
        ofstream fphi;
        ofstream fphiExact;
        ofstream fgrid;
        fphi.open("phi.csv");
        fphiExact.open("phi_exact.csv");
        fgrid.open("grid.csv");
        for (int i=0;i<Nq;i++){
            fphi << phi[i][0] << ",";
            fphiExact << phiExact(i)<<",";
            fgrid << gridX(i)<<","<<gridY(i)<<std::endl;
        }
        fphi.close();
    }
};

int main() {
//    Grid grid(2.0,2.0,100,100);
//    grid.poissonSolver();
//    grid.writeToCSV();
//    std::cout<<"e: " <<grid.error<<std::endl;

    int Ne = 100;
    Eigen::MatrixXd e(Ne,1);
    Eigen::MatrixXd n(Ne,1);
    int Nx = 0;
    int Ny = 0;
    for (int i=0;i<Ne;i++) {
        Nx+=5;
        Ny+=5;
        std::cout<<"i: "<<i<<std::endl;
        Grid grid(2.0,2.0,Nx,Ny);
        grid.poissonSolver();
        e(i) = grid.error;
        n(i) = Nx*Ny;
        if (i==Ne-1) {grid.writeToCSV();}
    }
    saveData("e.csv",e);
    saveData("n.csv",n);
    return 0;
}