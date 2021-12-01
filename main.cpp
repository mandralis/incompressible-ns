#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <eigen-io.h>
#include <fftw3.h>
#include <random>

#define _USE_MATH_DEFINES

using Eigen::MatrixXd;
using Eigen::Vector2d;

template <typename T> int sgn(T val)    {
    return (T(0) < val) - (val < T(0));
}

class Simulation {
public:
    // Physics
    double Re;
    // Nondimensionalization
    double length_scale, velocity_scale, time_scale;
    // Spatial discretization
    double Lx,Ly;
    double extentX,extentY;
    int Nx,Ny,Nq;
    double dx,dy;
    // Temporal discretization
    double T, dt;
    int Nt;
    // Velocity field and pressure field in row major order on staggered grid
    MatrixXd u,v,uT,vT;
    // Interpolated quantities (at cell corners)
    MatrixXd uq,vq;
    // Interpolated quantities (at cell centers)
    MatrixXd um,vm;
    // Spatial Derivatives
    MatrixXd div;
    MatrixXd omega,omegaT;
    MatrixXd lapU, lapV;
    MatrixXd convU, convV;
    MatrixXd duvdx, duvdy;
    MatrixXd duudx, dvvdy;
    MatrixXd dpdx,dpdy;
    // Time Derivatives
    MatrixXd dudt,dvdt,dudtT,dvdtT;
    // Grid and Poisson setup
    Eigen::MatrixXd gridX,gridY,gridXU,gridYU, gridXV, gridYV;
    Eigen::MatrixXd kX,kY;
    Eigen::MatrixXd kXmod,kYmod;
    // FFTW setup
    fftw_complex *f, *f_hat,*p, *p_hat;
    fftw_plan forward_plan, inverse_plan;

    // Constructor
    Simulation(double Re_, double U0_, double Lx_, double Ly_, int Nx_, int Ny_, int Nt_, double T_)
      : Re(Re_), Lx(Lx_),Ly(Ly_), Nx(Nx_),Ny(Ny_),Nt(Nt_),Nq(Nx_*Ny_),T(T_),
        u(Nq,1), v(Nq,1), uq(Nq,1), vq(Nq,1),lapU(Nq,1), lapV(Nq,1),
        duvdx(Nq,1), duvdy(Nq,1), duudx(Nq,1), dvvdy(Nq,1),
        convU(Nq,1), convV(Nq,1), dudt(Nq,1), dvdt(Nq,1), um(Nq,1), vm(Nq,1), div(Nq,1),omega(Nq,1),
        dpdx(Nq,1), dpdy(Nq,1), gridX(Nq,1), gridY(Nq,1), kX(Nq,1), kY(Nq,1), kXmod(Nq,1), kYmod(Nq,1),
        uT(Nq,1), vT(Nq,1),omegaT(Nq,1),dudtT(Nq,1),dvdtT(Nq,1),
        gridXU(Nq,1),gridYU(Nq,1), gridXV(Nq,1), gridYV(Nq,1) {

        // Define the non-dimensionalization scales
        length_scale = Lx;
        velocity_scale = U0_;
        time_scale = Lx/U0_;

        // Define the extent of the simulation domain
        extentX = (2*M_PI*Lx)/length_scale;
        extentY = (2*M_PI*Ly)/length_scale;

        // Compute the non-dimensionalized dx,dy,dt
        dx = (2*M_PI*Lx/Nx)/length_scale;
        dy = (2*M_PI*Ly/Ny)/length_scale;
        dt = (T/Nt)/time_scale;

        // Print information for the user
        std::cout<<"**************2D Navier Stokes sim **************" <<std::endl;
        std::cout<<"**************Simulation parameters**************" <<std::endl;
        std::cout<<"Re:"<<Re<<std::endl;
        std::cout<<"lengthscale:"<<length_scale<<std::endl;
        std::cout<<"velocityscale:"<<velocity_scale<<std::endl;
        std::cout<<"timescale:"<<time_scale<<std::endl;
        std::cout<<"dx:"<<dx<<std::endl;
        std::cout<<"dy:"<<dy<<std::endl;
        std::cout<<"dt:"<<dt<<std::endl;
        std::cout<<"**************Beginning simulation**************" <<std::endl;

        // Compute the grid coordinates and wavenumbers needed for the poisson equation
        computeGridCoordinates();
        writeGridCoordinates();

        // Allocate the memory required for fourier transforms
        f = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
        f_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
        p = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
        p_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);

        // Pre-define the fourier transforms using FFTW_MEASURE to maximize speed.
        forward_plan = fftw_plan_dft_2d(Nx, Ny, f, f_hat, FFTW_FORWARD, FFTW_MEASURE);
        inverse_plan = fftw_plan_dft_2d(Nx, Ny, p_hat, p, FFTW_BACKWARD, FFTW_MEASURE);
    }
    ~Simulation() {
        fftw_destroy_plan(forward_plan);
        fftw_destroy_plan(inverse_plan);
        fftw_cleanup();
    }

    // Grid methods
    void computeGridCoordinates() {
        // Grid goes from 0 <= x <= 2*pi-dx, 0 <= y <= 2*pi-dy
        // (gridX, gridY) are the lower left corners of each element of the grid
        // (gridXU,gridYU), (gridXV,gridYV) are the left face and bottom face of each element of the grid, respectively
        double Px = extentX;
        double Py = extentY;

        double y = extentY-dy;
        for (int j=0;j<Ny;j++) {
            double x = 0.0;
            for (int i=0;i<Nx;i++) {
                gridX(i+Nx*j) = x;
                gridY(i+Nx*j) = y;
                kX(i+Nx*j) = 2*M_PI/Px*i;
                kY(i+Nx*j) = 2*M_PI/Py*j;
                if(i>Nx/2){kX(i+Nx*j)=-(Nx-i)*2*M_PI/Px;}
                if(j>Ny/2){kY(i+Nx*j)=-(Ny-j)*2*M_PI/Py;}
                x+=dx;
                // Compute the modified wavenumbers
                kXmod(i+Nx*j) = 2/dx*sin(kX(i+Nx*j)*dx/2)*sgn(kX(i+Nx*j));
                kYmod(i+Nx*j) = 2/dy*sin(kY(i+Nx*j)*dy/2)*sgn(kY(i+Nx*j));
                // Write the staggered grids for easy imposition of initial conditions by the user
                gridXU(i+Nx*j) = x;
                gridYU(i+Nx*j) = y + dy/2;
                gridXV(i+Nx*j) = x + dx/2;
                gridYV(i+Nx*j) = y;
//                std::cout << "("<<gridX(i+Nx*j)<<","<<gridY(i+Nx*j)<<")"<<std::endl;
            }
            y-=dy;
        }
    }

    // Simulation methods
    void advance() {
        VectorXd k1u(Nq,1);
        VectorXd k2u(Nq,1);
        VectorXd k3u(Nq,1);
        VectorXd k1v(Nq,1);
        VectorXd k2v(Nq,1);
        VectorXd k3v(Nq,1);
        // Step 1
        computeRHS();
        k1u = dt * dudt;
        k1v = dt * dvdt;
        u = u + 0.5 * k1u;
        v = v + 0.5 * k1v;
        computeProjectionStep();
        computeDivergence(true);
        // Step 2
        computeRHS();
        k2u = dt * dudt;
        k2v = dt * dvdt;
        u = u - 1.5 * k1u + 2 * k2u;
        v = v - 1.5 * k1v + 2 * k2v;
        computeProjectionStep();
        computeDivergence(true);
        // Step 3
        computeRHS();
        k3u = dt * dudt;
        k3v = dt * dvdt;
        u = u + 7.0/6.0 * k1u -  4.0/3.0 * k2u + 1.0/6.0 * k3u;
        v = v + 7.0/6.0 * k1v -  4.0/3.0 * k2v + 1.0/6.0 * k3v;
        computeProjectionStep();
        computeDivergence(true);
        // Compute the vorticity for visualization
        computeVorticity();
    }

    void computeRHS()  {
        computeLaplacian();
        computeConvectiveTerm();
        dudt = -convU + lapU/Re;
        dvdt = -convV + lapV/Re;
    }

    void computeRHSPoisson() {
        computeDivergence();
        for (int i=0;i<Nq;i++) {
            f[i][0] = div(i)/dt;
            f[i][1] = 0.0;
        }
    }

    void computeProjectionStep() {
        solvePoissonEq();
        computePressureDerivatives();
        applyFractionalStep();
    }

    void solvePoissonEq() {
        computeRHSPoisson();
        // Compute forward transform of f
        fftw_execute(forward_plan);
        for (int i=1;i<Nq;i++){
            p_hat[i][0] = -f_hat[i][0] / (kXmod(i)*kXmod(i) + kYmod(i)*kYmod(i));
            p_hat[i][1] = -f_hat[i][1] / (kXmod(i)*kXmod(i) + kYmod(i)*kYmod(i));
        }
        // Set the fourier coefficient at (0,0) to an arbitrary value...it is just the mean
        p_hat[0][0] = 1.0;
        p_hat[0][1] = 1.0;
        // Compute the inverse fourier transform of p_hat
        fftw_execute(inverse_plan);
        // Normalize the elements of p by 1/(Nx*Ny)
        for (int i=0;i<Nq;i++) {
            p[i][0] = p[i][0]/(Nx*Ny);
            p[i][1] = p[i][1]/(Nx*Ny);
        }
    }

    void applyFractionalStep() {
        // Function that computes the divergence free velocity field given the pressure gradients
        u = u - dt*dpdx;
        v = v - dt*dpdy;
    }

    void computeDivergence(bool print=false) {
        // Function that computes divergence in cell centers since this is where the pressure is advanced
        int iX_, iXX, iY_, iYY;
        double meanDiv = 0.0;
        for (int i = 0; i < Nq; i++) {
            getAdjacentIndices(i, iX_, iXX, iY_, iYY);
            div(i) = 1/dx * (u(iXX) - u(i)) + 1/dy * (v(iY_) - v(i));
            meanDiv += abs(div(i));
        }
        meanDiv /= Nq;
        if (print) {std::cout << "Mean divergence after projection is: " << meanDiv << std::endl;}
    }

    void computeVorticity() {
        // compute the vorticity at the center of the cells
        interpUVCenter();
        int iX_,iXX,iY_,iYY;
        for (int i=0;i<Nq;i++) {
            getAdjacentIndices(i, iX_,iXX,iY_,iYY);
            omega(i) = 1/(2*dx) * (vm(iXX)-vm(iX_)) - 1/(2*dy) * (um(iY_)-um(iYY));
        }
    }

    void computePressureDerivatives() {
        // Function that computes pressure derivatives at cell centers to apply the fractional step
        int iX_,iXX,iY_,iYY;
        for (int i=0;i<Nq;i++) {
            getAdjacentIndices(i, iX_,iXX,iY_,iYY);
            dpdx(i) = 1/dx * (p[i][0]-p[iX_][0]);
            dpdy(i) = 1/dy * (p[i][0]-p[iYY][0]);
        }
    }

    void computeLaplacian() {
        // Function that computes Laplacian on grid faces
        int iX_,iXX,iY_,iYY;
        for (int i=0;i<Nq;i++) {
            getAdjacentIndices(i, iX_,iXX,iY_,iYY);
            lapU(i) = 1/(dx*dx) * (u(iXX)-2*u(i)+u(iX_)) +  1/(dy*dy) * (u(iY_)-2*u(i)+u(iYY));
            lapV(i) = 1/(dx*dx) * (v(iXX)-2*v(i)+v(iX_)) +  1/(dy*dy) * (v(iY_)-2*v(i)+v(iYY));
        }
    }

    void computeConvectiveTerm() {
        interpUVFaceToCorner();
        computeCrossTerm();
        computeSquareTerm();
        convU = duudx + duvdy;
        convV = duvdx + dvvdy;
    }

    void computeSquareTerm() {
        // Function that computes duudx and dvvdy on grid faces
        int iX_,iXX,iY_,iYY;
        for (int i=0;i<Nq;i++) {
            getAdjacentIndices(i, iX_,iXX,iY_,iYY);
            duudx(i) = 1/(2*dx) * (u(iXX)*u(iXX)-u(iX_)*u(iX_));
            dvvdy(i) = 1/(2*dy) * (v(iY_)*v(iY_)-v(iYY)*v(iYY));
        }
    }

    void computeCrossTerm() {
        // Function that computes duvdx and duvdy on grid faces
        int iX_,iXX,iY_,iYY;
        for (int i=0;i<Nq;i++) {
            getAdjacentIndices(i, iX_,iXX,iY_,iYY);
            duvdy(i) = 1/dy * (uq(iY_)*vq(iY_)-uq(i)*vq(i));
            duvdx(i) = 1/dx * (uq(iXX)*vq(iXX)-uq(i)*vq(i));
        }
    }

    // Interpolation methods
    void interpUVCenter() {
        // Function that linearly interpolates the grid velocities from the left and bottom face to
        // the center
        int iX_,iXX,iY_,iYY;
        for (int i=0;i<Nq;i++) {
            getAdjacentIndices(i, iX_,iXX,iY_,iYY);
            um(i) = 0.5 * (u(i) + u(iXX));
            vm(i) = 0.5 * (v(i) + v(iY_));
        }
    }

    void interpUVFaceToCorner() {
        // Function that linearly interpolates the grid velocities from the left and bottom face to
        // the lower left corner
        int iX_,iXX,iY_,iYY;
        for (int i=0;i<Nq;i++) {
            getAdjacentIndices(i, iX_,iXX,iY_,iYY);
            uq(i) = 0.5 * (u(i) + u(iYY));
            vq(i) = 0.5 * (v(i) + v(iX_));
        }
    }

    // Periodic boundary conditions methods
    void getAdjacentIndices(const int &i, int &iX_,int &iXX,int &iY_,int &iYY) {
        iX_ = i-1;
        iXX = i+1;
        if(i%Nx==0){iX_=iX_+Nx;}
        if((i+1)%Nx==0) {iXX=iXX-Nx;}
        iY_ = i-Nx;
        iYY = i+Nx;
        if(iY_<0) {iY_=iY_+Nq;}
        if(iYY>Nq-1) {iYY=iYY-Nq;}
    }

    // Initial conditions
    void setTaylorGreenIC(double A, double a, double B, double b) {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<double> dist(-0.1,0.1); // distribution in range [-1, 1]
        int count = 0;
        double xU,yU,xV,yV;
        for (int j=0;j<Nx;j++) {
            for (int i=0;i<Ny;i++) {
                xU = gridXU(count);
                yU = gridYU(count);
                xV = gridXV(count);
                yV = gridYV(count);
                u(count) = A*sin(a*xU)*cos(b*yU);
                v(count) = B*cos(a*xV)*sin(b*yV);
//                if (i>Nx/2+7 and i < Nx/2+17 and j>Ny/2+7 and j<Ny/2+17) {
//                    u(count) = A*sin(a*xU)*cos(b*yU) + 0.1*abs(A);
//                    v(count) = B*cos(a*xV)*sin(b*yV) + 0.1*abs(B);
//                }
                // Non-dimensionalize the velocity !
                u(count)/=velocity_scale;
                v(count)/=velocity_scale;
                count++;
            }
        }
        computeVorticity();
    }

    void setVorticityIC() {
        // Random number generator
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<double> angle(0.0,2*M_PI); // uniform distribution in range [0, 2pi]

        // FFTW setup
        fftw_complex *omega_hat, *psi, *psi_hat;
        fftw_plan inverse_plan_psi;
        omega_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
        psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
        psi_hat = (fftw_complex*)   fftw_malloc(sizeof(fftw_complex) * Nq);
        inverse_plan_psi = fftw_plan_dft_2d(Nx, Ny, psi_hat, psi, FFTW_BACKWARD, FFTW_ESTIMATE);

        // Initialize the fourier coefficients of vorticity with equal magnitude 1 but arbitrary phases.
        // Excite all wavenumbers, change this later
        double M = 1000.0;
        int count = 0;
        for (int j=0;j<Nx;j++) {
            for (int i=0;i<Ny;i++) {
                double theta = angle(rng);
                if (Nx/4<=j<=3*Nx/4 and Ny/4<=j<=3*Ny/4 ) {
                    omega_hat[count][0] = M*cos(theta);
                    omega_hat[count][1] = M*sin(theta);
                } else {
                    omega_hat[count][0] = 0.0;
                    omega_hat[count][1] = 0.0;
                }
                count++;
            }
        }
//        for (int i=0;i<Nq;i++){
////            double theta = angle(rng);
//            omega_hat[i][0] = M*cos(angle(rng));
//            omega_hat[i][1] = M*sin(angle(rng));
//        }
        // Compute psi_hat
        for (int i=1;i<Nq;i++){
            psi_hat[i][0] = omega_hat[i][0] / (kX(i)*kX(i) + kY(i)*kY(i));
            psi_hat[i][1] = omega_hat[i][1] / (kX(i)*kX(i) + kY(i)*kY(i));
        }
        // Set the fourier coefficient at (0,0) to an arbitrary value...it is just the mean
        psi_hat[0][0] = 0.0;
        psi_hat[0][1] = 0.0;
        // Compute the inverse fourier transform of psi_hat
        fftw_execute(inverse_plan_psi);
        // Normalize the elements of psi by 1/(Nx*Ny)
        for (int i=0;i<Nq;i++) {
            psi[i][0] = psi[i][0]/(Nx*Ny);
            psi[i][1] = psi[i][1]/(Nx*Ny);
        }
        // Finally the velocity field is computed by finite differencing the stream function psi
        int iX_,iXX,iY_,iYY;
        for (int i=0;i<Nq;i++) {
            getAdjacentIndices(i, iX_,iXX,iY_,iYY);
            u(i) = 1/dy * (psi[i][0]-psi[iYY][0]);
            v(i) = -1/dx * (psi[i][0]-psi[iX_][0]);
            // Non dimensionalize !
            u(i) /= velocity_scale;
            v(i) /= velocity_scale;
        }
        computeVorticity();
    }

    void setImpingingVortexIC() {
    // Random number generator
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> angle(0.0,2*M_PI); // uniform distribution in range [0, 2pi]

    // FFTW setup
    fftw_complex *omega_, *omega_hat, *psi, *psi_hat;
    fftw_plan forward_plan_omega, inverse_plan_psi;
    omega_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
    omega_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
    psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
    psi_hat = (fftw_complex*)   fftw_malloc(sizeof(fftw_complex) * Nq);
    forward_plan_omega = fftw_plan_dft_2d(Nx, Ny, omega_, omega_hat, FFTW_FORWARD, FFTW_ESTIMATE);
    inverse_plan_psi = fftw_plan_dft_2d(Nx, Ny, psi_hat, psi, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Set the vorticity field
    // Vortex center
    double vortexX = 3*M_PI/4;
    double vortexY = M_PI;
    double vortexR = M_PI/4;
    double Gamma0 = 1;
    double sigma = 0.21*vortexR;
    int count = 0;
    double x,y;
    for (int j=0;j<Nx;j++) {
        for (int i=0;i<Ny;i++) {
            x = gridX(count);
            y = gridY(count);
            // Compute distance from vortex center
            double d = vortexR - sqrt((x-vortexX)*(x-vortexX) + (y-vortexY)*(y-vortexY));
            if (d>=0.0) {
                omega_[count][0] = Gamma0/M_PI/(sigma*sigma)*exp(-d/(sigma*sigma));
                omega_[count][1] = 0.0;
            } else {
                omega_[count][0] = 0.0;
                omega_[count][1] = 0.0;
            }
            count++;
        }
    }

    // Fourier transform the vorticity field
    fftw_execute(forward_plan_omega);

    // Compute psi_hat
    for (int i=1;i<Nq;i++){
        psi_hat[i][0] = omega_hat[i][0] / (kX(i)*kX(i) + kY(i)*kY(i));
        psi_hat[i][1] = omega_hat[i][1] / (kX(i)*kX(i) + kY(i)*kY(i));
    }
    // Set the fourier coefficient at (0,0) to an arbitrary value...it is just the mean
    psi_hat[0][0] = 0.0;
    psi_hat[0][1] = 0.0;
    // Compute the inverse fourier transform of psi_hat
    fftw_execute(inverse_plan_psi);
    // Normalize the elements of psi by 1/(Nx*Ny)
    for (int i=0;i<Nq;i++) {
        psi[i][0] = psi[i][0]/(Nx*Ny);
        psi[i][1] = psi[i][1]/(Nx*Ny);
    }
    // Finally the velocity field is computed by finite differencing the stream function psi
    int iX_,iXX,iY_,iYY;
    for (int i=0;i<Nq;i++) {
        getAdjacentIndices(i, iX_,iXX,iY_,iYY);
        u(i) = 1/dy * (psi[i][0]-psi[iYY][0]);
        v(i) = -1/dx * (psi[i][0]-psi[iX_][0]);
        // Non dimensionalize !
//        u(i) /= velocity_scale;
//        v(i) /= velocity_scale;
    }
    // Set some fluid moving the right just before the vortex ring
    std::uniform_real_distribution<double> dist(-0.03,0.03); // distribution in range [-1, 1]
    count = 0;
    for (int j=0;j<Nx;j++) {
        for (int i=0;i<Ny;i++) {
            x = gridX(count);
            if (x<M_PI/2) {
                u(count)=0.5 + dist(rng);
                v(count)=0.0;
            }
            // Non-dimensionalize the velocity !
            u(count)/=velocity_scale;
            v(count)/=velocity_scale;
            count++;
        }
    }
    computeVorticity();
}

    void setMixingLayerIC(double fraction_still) {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<double> dist(-0.03,0.03); // distribution in range [-1, 1]
        int count = 0;
        double x,y;
        for (int j=0;j<Nx;j++) {
            for (int i=0;i<Ny;i++) {
                x = gridX(count);
                y = gridY(count);
                if(y>extentY/2.0 + extentY*fraction_still/2) {
                    u(count) = 2*(extentY-y)/extentY + dist(rng);
                    v(count) = 0.0;
                } else if (y<=extentY/2.0 - extentY*fraction_still/2) {
                    u(count) = -2*y/extentY + dist(rng);
                    v(count) = 0.0;
                } else {
                    u(count) = 0.0;
                    v(count) = 1.0;
                }
                // Non-dimensionalize the velocity !
                u(count)/=velocity_scale;
                v(count)/=velocity_scale;
                count++;
            }
        }
        computeVorticity();
    }

    void setTestStaggeredGridIC() {
        double xU,yU,xV,yV;
        for (int i=0;i<Nq;i++) {
            xU = gridXU(i);
            yU = gridYU(i);
            xV = gridXV(i);
            yV = gridYV(i);
            u(i) = cos(xU);
            v(i) = sin(yV);
        }
    }

    void setRandomIC() {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<double> dist(-1.0,1.0); // distribution in range [-1, 1]
        double x,y;
        for (int i=0;i<Nq;i++) {
            u(i) = dist(rng)/velocity_scale;
            v(i) = dist(rng)/velocity_scale;
        }
    }

    // Testing methods
    void computeStaggeredGridError() {
        double xU,yU,xV,yV;
        double error = 0.0;
        for (int i=0;i<Nq;i++) {
            xU = gridXU(i);
            yU = gridYU(i);
            xV = gridXV(i);
            yV = gridYV(i);
            dudtT(i) = 2*sin(xU)*cos(xU) - cos(xU)*cos(yU)   -cos(xU)/Re;
            dvdtT(i) = sin(xV)*sin(yV)   - 2*cos(yV)*sin(yV) -sin(yV)/Re;
            error += abs(dudtT(i)-dudt(i)) + abs(dvdtT(i)-dvdt(i));
        }
        error/=Nq;
        std::cout<<"error: "<<error<<std::endl;
    }

    double computeTaylorGreenError(double t) {
        double x,y,xU,yU,xV,yV;
        double error = 0.0;
        for (int i=0;i<Nq;i++) {
            x = gridX(i);
            y = gridY(i);
            xU = gridXU(i);
            yU = gridYU(i);
            xV = gridXV(i);
            yV = gridYV(i);
            uT(i) = sin(xU)*cos(yU)*exp(-2*t/Re);
            vT(i) = -cos(xV)*sin(yV)*exp(-2*t/Re);
            omegaT(i) = 2*sin(x+dx)*sin(y+dy)*exp(-2*t/Re);
            error += abs(uT(i)-u(i)) + abs(vT(i)-v(i));
        }
        error/=Nq;
        std::cout<<"error: "<<error<<std::endl;
        return error;
    }

    // IO methods
    void writeUV(const int &iter) {
        // Writes the solution u,v to a CSV file
        ofstream fu,fv,fomega,fp;
        fu.open("u_" + std::to_string(iter) + ".csv");
        fv.open("v_" + std::to_string(iter) + ".csv");
        fomega.open("omega_" + std::to_string(iter) + ".csv");
//        fp.open("p_" + std::to_string(iter) + ".csv");
        for (int i=0;i<Nq;i++){
            fu << u(i) <<",";
            fv << v(i)<<",";
            fomega << omega(i)<<",";
//            fp << p[i][0]<<",";
        }
        fu.close();
        fv.close();
        fomega.close();
    }

    void writeDiv() {
        // Writes the divergence to a CSV file
        ofstream fdiv;
        fdiv.open("div.csv",std::ios_base::app);
        double meanDiv = 0.0;
        for (int i=0;i<Nq;i++){
            meanDiv+=abs(div(i));
        }
        meanDiv/=Nq;
        fdiv << meanDiv <<",";
        fdiv.close();
    }

    void writeGridCoordinates() {
        // Writes the gridCoordinates
        ofstream fgrid,fgridU,fgridV;
        fgrid.open("grid.csv");
        fgridU.open("gridU.csv");
        fgridV.open("gridV.csv");
        for (int i=0;i<Nq;i++){
            fgrid << gridX(i)<<","<<gridY(i)<<std::endl;
            fgridU << gridXU(i)<<","<<gridYU(i)<<std::endl;
            fgridV << gridXV(i)<<","<<gridYV(i)<<std::endl;
        }
        fgrid.close();
    }
};

int main() {
    // There are two free parameters (Re,Lx)
    // Set the viscosity to 1.0 -->
    // Calculate the reference velocity in the simulation class (U0=Re/Lx)

    // Choose U0 to be the maximum of the initial condition - detect this in the class constructor
    // Once U0 is found, since Lx and Re will be set by the user, adapt the viscosity to match this

    // Non-dimensionalization
    double Re = 1000;
    double U0 = 1.0;
    double Lx = 1.0;

    // Grid
    int Nx = 256;

    // Time
    double T = 10.0;
    int Nt = 1000;

    // Square, uniform grid
    double Ly = Lx;
    int Ny = Nx;

    Simulation sim(Re, U0, Lx, Ly, Nx, Ny, Nt, T);

//    sim.setRandomIC();
    double A,a,B,b;
    A=1.0;a=1.0;B=-1.0;b=1.0;
    sim.setTaylorGreenIC(A,a,B,b);
//    sim.setVorticityIC();
//    sim.setMixingLayerIC(0.0);
//    sim.setImpingingVortexIC();
    sim.writeUV(0);
    sim.computeTaylorGreenError(0.0 );

    double t = sim.dt;
    double globalError = 0.0;
    double error = 0.0;
    for (int i=1;i<Nt;i++) {
        std::cout<<"iter="<<i<<std::endl;
        sim.advance();
        sim.writeUV(i);
//        sim.writeDiv();
        error = sim.computeTaylorGreenError(t);
        t += sim.dt;
        globalError += error;
    }
    globalError/=Nt;
    std::cout<<"Global error is: "<<globalError<<std::endl;
    return 0;
}

// TESTING STUFF
// ALL INTERPOLATION FUNCTIONS WORK
//    sim.setTestIC();
//    // Test interpolation functions
//    sim.interpCrossTermCornerToFace();

//  STAGGERED GRID DIFFERENTIATION IS ORDER 2 ACCURATE ????? --> YES !
//    sim.setTestStaggeredGridIC();
//    sim.computeRHS();
//    sim.computeStaggeredGridError();

// FRACTIONAL STEP METHOD WORKS
//    sim.setTestRandomIC();
//    for (int i=1;i<Nt;i++) {
//        sim.advance();
//    }


//// Spring mass system RHS function
//Vector2d compute_rhs_springmass(const Vector2d &u) {
//    double m = 1.0; // careful...hardcoded
//    double k = 1.0;
//    Vector2d dudt(0.0,0.0);
//    dudt(0) = u(1);
//    dudt(1) = -k/m * u(0);
//    return dudt;
//};
//
//Vector2d sol_springmass(const double t) {
//    Vector2d u(0.0,0.0);
//    u(0) = sqrt(2) * cos(t - M_PI/4.0);
//    u(1) = -sqrt(2) * sin(t - M_PI/4.0);
//    return u;
//}
//
//// Van der pol oscillator RHS function
//Vector2d compute_rhs_van(const Vector2d &u) {
//    double mu = 1.0; // careful...hardcoded
//    Vector2d dudt(0.0,0.0);
//    dudt(0) = u(0);
//    dudt(1) = (mu*(1.0-u(0)*u(0))*u(1)-u(0));
//    return dudt;
//};
//
//// Time advancing function
//Vector2d time_advance_RK3(const Vector2d &u0, const double &dt) {
//    Vector2d u(0.0,0.0);
//    // Step 1
//    Vector2d F = compute_rhs_springmass(u0);
//    Vector2d k1 = dt * F;
//    // Step 2
//    F = compute_rhs_springmass(u0 + 0.5*k1);
//    Vector2d k2 = dt * F;
//    // Step 3
//    F = compute_rhs_springmass(u0 - k1 + 2*k2);
//    Vector2d k3 = dt * F;
//    // Time advancement
//    u = u0 + 1.0/6.0 * k1 + 2.0/3.0 * k2 + 1.0/6.0 * k3;
//    return u;
//}
//
//// Time advancing function
//Vector2d time_advance_euler(const Vector2d &u0, const double &dt) {
//    Vector2d u(0.0,0.0);
//    // Time advancement
//    u = u0 + dt * compute_rhs_springmass(u0);
//    return u;
//}


//    double T = 15.0;
//    // Test order of convergence
//    int Nconv = 100;
//    // Error matrix
//    MatrixXd e_mat = MatrixXd::Zero(1,Nconv);
//    // N matrix
//    MatrixXd N_mat = MatrixXd::Zero(1,Nconv);
//
//    for (int i=1;i<Nconv;i++) {
//        std::cout << "N: " << i << endl;
//        int Nt = 100*i;
//        double dt = T/Nt;
//        Vector2d u0(1.0,1.0);
//
//        MatrixXd u_sol = MatrixXd::Zero(2,Nt+1);
//        MatrixXd u_hat = MatrixXd::Zero(2,Nt+1);
//        MatrixXd t_vec = MatrixXd::Zero(1,Nt+1);
//        u_hat.block<2,1>(0,0) = u0;
//        u_sol.block<2,1>(0,0) = u0;
//
//        // Initialize the time
//        double t = 0.0 + dt;
//        // Initialize the error
//        double e = 0.0;
//        for (int j=1;j<=Nt;j++) {
//            // Advance the simulation
//            Vector2d u = time_advance_RK3(u0,dt);
//            Vector2d u_exact = sol_springmass(t);
//            // Write to approximate solution matrix
//            u_hat.block<2,1>(0,j) = u;
//            // Write to exact solution matrix
//            u_sol.block<2,1>(0,j) = u_exact;
//            // Write to time matrix
//            t_vec(0,j) = t;
//            // Update current velocity
//            u0 = u;
//            // Update the current time
//            t += dt;
//            // Compute the error
//            e += pow(u(0,0)-u_exact(0,0),2.0) + pow(u(1,0)-u_exact(1,0),2.0);
//        }
//        // Store the values of the error
//        e_mat(0,i) = sqrt(e/Nt);
//        // Store the values of N
//        N_mat(0,i) = Nt;
//        saveData("u_hat.csv",u_hat);
//        saveData("u_sol.csv",u_sol);
//        saveData("t.csv",t_vec);
//    }
//    saveData("e.csv",e_mat);
//    saveData("N.csv",N_mat);



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