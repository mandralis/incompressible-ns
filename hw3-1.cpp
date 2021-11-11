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

// Tridiagonal solver
void tridiag(Eigen::MatrixXd &a, Eigen::MatrixXd &b, Eigen::MatrixXd &c, Eigen::MatrixXd &r, Eigen::MatrixXd &u)
{
    int j,n=a.rows();
    double bet;
    Eigen::MatrixXd gam(n,1);
    if (b(0) == 0.0) throw ("Error 1 in tridiag");
    u(0)=r(0)/(bet=b(0));
    for (j=1;j<n;j++) {
        gam(j)=c(j-1)/bet;
        bet=b(j)-a(j)*gam(j);
        if (bet == 0.0) throw("Error 2 in tridiag");
        u(j)=(r(j)-a(j)*u(j-1))/bet;
    }
    for (j=(n-2);j>=0;j--)
        u(j) -= gam(j+1)*u(j+1);
}

class Grid {
public:
    double Lx,Ly,nu,betaX,betaY,dt,dx,dy;
    int Nx,Ny,N,Nint,Nxint,Nyint;
    Eigen::MatrixXd gridX,gridY,u,v,fU,fV;
    Grid(double Lx_, double Ly_, int Nx_, int Ny_, double nu_,double dt_) :
    Lx(Lx_),Ly(Ly_),Nx(Nx_),Ny(Ny_), N(Nx_*Ny_), Nint((Nx_-1)*(Ny_-1)), gridX(N,1),gridY(N,1),
    Nxint(Nx-1),Nyint(Ny-1),u(N,1), v(N,1), fU(Nint,1),fV(Nint,1),nu(nu_),dt(dt_){
        // Define dx and dy
        dx = Lx/(Nx-1);
        dy = Ly/(Ny-1);
        std::cout<<"dx: " <<dx<<", dy: "<<dy<<std::endl;
        // Define beta
        betaX = nu*dt/(2*dx*dx);
        betaY = nu*dt/(2*dy*dy);
        // Compute the grid coordinates in the constructor
        computeGridCoordinates();
        initializeVelocityField();
        std::cout<<"Constructor done"<<std::endl;
    }

    void computeGridCoordinates() {
        // Grid goes from 0 <= x <= 1, 0 <= y <= 1
        // Everything in row major order
        double y = Ly;
        for (int j=0;j<Ny;j++) {
            double x = 0.0;
            for (int i=0;i<Nx;i++) {
                gridX(i+Nx*j) = x;
                gridY(i+Nx*j) = y;
                x+=dx;
            }
            y-=dy;
        }
    }

    void initializeVelocityField() {
        for (int i=0;i<N;i++) {
            double x = gridX(i);
            double y = gridY(i);
            u(i) = sin(2*M_PI*x);
            v(i) = 1 - y;
            // Specialize cases to incorporate the boundaries
            if (x<1e-14){
                u(i) = 0.0;
                v(i) = 1-y;
            }
            if (y>1.0-1e-14){
                u(i) = sin(2*M_PI*x);
                v(i) = 0.0;
            }
            if (x>1.0-1e-14){
                u(i) = 0.0;
                v(i) = 1-y;
            }
            if (y<1e-14){
                u(i) = sin(2*M_PI*x);
                v(i) = 1.0;
            }
        }
    }

    void computeRHS() {
        // Everything in row major order
        // First solve for xi
        Eigen::MatrixXd xiU(N,1), xiV(N,1);
        int n = 0;
        for (int j=0;j<Ny;j++) {
            for (int i=0;i<Nx;i++) {
                double y = gridY(n);
                if (y<1e-14 or y>1.0-1e-14){xiU(n)=0.0;xiV(n)=0.0;} // top and bottom boundaries: zero padding, xi doesn't exist here
                else {
                    xiU(n) = u(n) + betaY * (u(n-Nx) - 2*u(n) + u(n+Nx)); // this works everywhere because we already avoided the edges
                    xiV(n) = v(n) + betaY * (v(n-Nx) - 2*v(n) + v(n+Nx)); // this works everywhere because we already avoided the edges
                }
                n++;
            }
        }
        // Then solve for a (also solve for b, the euler part, in the same loop)
        Eigen::MatrixXd aU(N,1), aV(N,1);
        Eigen::MatrixXd bU(N,1), bV(N,1);
        n = 0;
        int nint = 0;
        for (int j=0;j<Ny;j++) {
            for (int i=0;i<Nx;i++) {
                double x = gridX(n);
                double y = gridY(n);
                if (x<1e-14 or y>1.0-1e-14 or x>1.0-1e-14 or y<1e-14){aU(n)=0.0;bU(n)=0.0;aV(n)=0.0;bV(n)=0.0;} // all boundaries: a(n) and b(n) only exist in the interior
                else {
                    aU(n) = xiU(n) + betaX * (xiU(n+1) - 2*xiU(n) + xiU(n-1)); // this works everywhere because we already avoided all undefined locations
                    bU(n) = dt * (u(n) * (u(n+1)-u(n-1))/(2*dx) + v(n) * (u(n-Nx)-u(n+Nx))/(2*dy));
                    aV(n) = xiV(n) + betaX * (xiV(n+1) - 2*xiV(n) + xiV(n-1)); // this works everywhere because we already avoided all undefined locations
                    bV(n) = dt * (u(n) * (v(n+1)-v(n-1))/(2*dx) + v(n) * (v(n-Nx)-v(n+Nx))/(2*dy));
                    fU(nint) = aU(n) + bU(n);
                    fV(nint) = aV(n) + bV(n);
                    nint++;
                }
                n++;
            }
        }
    }

    void advance() {
        computeRHS();
        // Define the vectors a,b,c of the tridiagonal matrix A
        Eigen::MatrixXd a = MatrixXd::Constant(Nx-1, 1, -betaX);
        Eigen::MatrixXd b = MatrixXd::Constant(Nx-1, 1, 1+2*betaX);
        Eigen::MatrixXd c = MatrixXd::Constant(Nx-1, 1, -betaX);
        Matrix<double,Dynamic,Dynamic,RowMajor> zU(Ny-1,Nx-1),zV(Ny-1,Nx-1);
        Eigen::MatrixXd ziU(Nx-1,1),ziV(Nx-1,1);
        for (int i=0;i<Ny-1;i++) {
            Eigen::MatrixXd fUi(Nx-1,1);
            fUi = fU.block(i*(Nx-1),0,Nx-1,1);
            tridiag(a, b, c, fUi, ziU);
            zU.block(i,0,1,Nx-1) = ziU.transpose();
            Eigen::MatrixXd fVi(Nx-1,1);
            fVi = fV.block(i*(Nx-1),0,Nx-1,1);
            tridiag(a, b, c, fVi, ziV);
            zV.block(i,0,1,Nx-1) = ziV.transpose();
        }

        // Solve for columns of z of length Ny
        Eigen::MatrixXd a1 = MatrixXd::Constant(Ny-1, 1, -betaY);
        Eigen::MatrixXd b1 = MatrixXd::Constant(Ny-1, 1, 1+2*betaY);
        Eigen::MatrixXd c1 = MatrixXd::Constant(Ny-1, 1, -betaY);
        Matrix<double,Dynamic,Dynamic,RowMajor> u_advanced(Ny-1,Nx-1), v_advanced(Ny-1,Nx-1);
        Eigen::MatrixXd ui(Ny-1,1), vi(Ny-1,1);
        for (int i=0;i<Nx-1;i++) {
            Eigen::MatrixXd zUvec(Ny-1,1);
            zUvec = zU.block(0,i,Ny-1,1);
            tridiag(a1, b1, c1, zUvec, ui);
            u_advanced.block(0,i,Ny-1,1) = ui;
            Eigen::MatrixXd zVvec(Ny-1,1);
            zVvec = zV.block(0,i,Ny-1,1);
            tridiag(a1, b1, c1, zVvec, vi);
            v_advanced.block(0,i,Ny-1,1) = vi;
        }

        // Put the matrices u,v back into row major form and into the member variables u, and v
        int nint = 0;
        for (int i=0;i<N;i++) {
            double x = gridX(i);
            double y = gridY(i);
            if (x<1e-14 or y>1.0-1e-14 or x>1.0-1e-14 or y<1e-14){}//do nothing if on the boundaries
            else { // fill all the interior points of u and v
                u(i) = u_advanced(nint);
                v(i) = v_advanced(nint);
//                std::cout<<"("<<u(i)<<","<<u(i)<<")"<<std::endl;
                nint++;
            }
            std::cout<<"("<<u(i)<<","<<u(i)<<")"<<std::endl;
        }
    }

    void writeToCSV() {
        // Writes the solution (u,v) to a CSV file
        ofstream fgrid;
        ofstream fu;
        ofstream fv;
        fgrid.open("grid.csv");
        fu.open("u.csv");
        fv.open("v.csv");
        for (int i=0;i<N;i++){
            fgrid << gridX(i)<<","<<gridY(i)<<std::endl;
            fu << u(i)<<","<<std::endl;
            fv << v(i)<<","<<std::endl;
        }
        fgrid.close();
    }
};

int main() {
    Grid grid(1.0,1.0,50,50,0.015,0.02);
//    grid.advance();
//    grid.writeToCSV();
    for(int i=0;i<1000;i++) {
        std::cout<<"i: "<<i<<std::endl;
        grid.advance();
        grid.writeToCSV();
    }
    return 0;
}