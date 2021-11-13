#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <eigen-io.h>
#include <fftw3.h>

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
    // Spatial discretization
    double Lx,Ly;
    int Nx,Ny;
    int Nxs,Nys,Nq;
    double dx,dy;
    // Temporal discretization
    double T,dt;
    int Nt;
    // Velocity field and pressure field in row major order on staggered grid
    MatrixXd u,v;
    // Interpolated quantities (at cell corners)
    MatrixXd uq,vq;
    // Interpolated quantities (at cell centers)
    MatrixXd um,vm;
    // Spatial Derivatives
    MatrixXd div;
    MatrixXd lapU, lapV;
    MatrixXd convU, convV;
    MatrixXd duvdx, duvdy;
    MatrixXd duvdxC, duvdyC;
    MatrixXd duudx, dvvdy;
    MatrixXd dpdx,dpdy;
    // Time Derivatives
    MatrixXd dudt,dvdt;
    // Grid and Poisson setup
    Eigen::MatrixXd gridX,gridY;
    Eigen::MatrixXd kX,kY;
    Eigen::MatrixXd kXmod,kYmod;
    // FFTW setup
    fftw_complex *f, *f_hat,*p, *p_hat;
    fftw_plan forward_plan, inverse_plan;

    // Constructor
    Simulation(double Re_, double Lx_, double Ly_, double T_, int Nx_, int Ny_, int Nt_)
      : Re(Re_), Lx(Lx_),Ly(Ly_),T(T_), Nx(Nx_),Ny(Ny_),Nt(Nt_),dx(Lx_/Nx_),dy(Ly_/Ny_),dt(T_/Nt_),Nxs(Nx_-1),Nys(Ny_-1),
        Nq(Nxs*Nys), u(Nq,1), v(Nq,1), uq(Nq,1), vq(Nq,1),lapU(Nq,1), lapV(Nq,1),
        duvdx(Nq,1), duvdy(Nq,1), duvdxC(Nq,1), duvdyC(Nq,1), duudx(Nq,1), dvvdy(Nq,1),
        convU(Nq,1), convV(Nq,1), dudt(Nq,1), dvdt(Nq,1), um(Nq,1), vm(Nq,1), div(Nq,1),
        dpdx(Nq,1), dpdy(Nq,1), gridX(Nq,1), gridY(Nq,1){
        std::cout << "Constructor" << std::endl;
        // Compute the grid coordinates and wavenumbers needed for the poisson equation
        computeGridCoordinates();
        // Don't do in place transforms...requires extra considerations
        f = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
        f_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
        p = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
        p_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
        forward_plan = fftw_plan_dft_2d(Nx, Ny, f, f_hat, FFTW_FORWARD, FFTW_MEASURE);
        inverse_plan = fftw_plan_dft_2d(Nx, Ny, p_hat, p, FFTW_BACKWARD, FFTW_MEASURE);
        std::cout << "Constructor done" << std::endl;
    }
    ~Simulation() {
        fftw_destroy_plan(forward_plan);
        fftw_destroy_plan(inverse_plan);
        fftw_cleanup();
    }

    void advance() {
        VectorXd k1u(Nq,1);
        VectorXd k2u(Nq,1);
        VectorXd k3u(Nq,1);
        VectorXd k1v(Nq,1);
        VectorXd k2v(Nq,1);
        VectorXd k3v(Nq,1);
        // Step 1
        std::cout << "computeRHS" << std::endl;
        computeRHS();
        std::cout << "computeRHS done" << std::endl;
        k1u = dt * dudt;
        k1v = dt * dvdt;
        u = u + 0.5 * k1u;
        v = v + 0.5 * k1v;
        std::cout << "computeProjectionStep" << std::endl;
        computeProjectionStep();
        std::cout << "computeProjectionStep done" << std::endl;
        // Step 2
        computeRHS();
        k2u = dt * dudt;
        k2v = dt * dvdt;
        u = u - 1.5 * k1u + 2 * k2u;
        v = v - 1.5 * k1v + 2 * k2v;
        computeProjectionStep();
        // Step 3
        computeRHS();
        k3u = dt * dudt;
        k3v = dt * dvdt;
        u = u + 7.0/6.0 * k1u -  4.0/3.0 * k2u + 1.0/6.0 * k3u;
        v = v + 7.0/6.0 * k1v -  4.0/3.0 * k2v + 1.0/6.0 * k3v;
        computeProjectionStep();
    }

    void computeGridCoordinates() {
        // Grid goes from -Lx/2 <= x <= Lx/2, -Ly/2 <= y <= Ly/2
        std::cout << "computeGridCoordinates" << std::endl;
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
        std::cout << "computeGridCoordinatesDone" << std::endl;
    }

    void computeRHSPoisson() {
        computeDivergence();
        for (int i=0;i<Nq;i++) {
            f[i][0] = dt * div(i);
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
        u = u - dt * dpdx;
        v = v - dt * dpdy;
    }

    void computeRHS() {
        computeLaplacian();
        computeConvectiveTerm();
        dudt = -convU + lapU/Re;
        dvdt = -convV + lapV/Re;
    }

    void computeDivergence() {
        // Function that computes divergence in cell centers
        // since this is where the pressure is advanced
        interpUVCenter();
        int iX_,iXX,iY_,iYY;
        for (int i=0;i<Nq;i++) {
            getAdjacentIndices(i, iX_,iXX,iY_,iYY);
            div(i) = 1/(2*dx) * (u(iXX)-u(iX_)) + 1/(2*dy) * (v(iYY)-v(iY_));
        }
    }

    void computePressureDerivatives() {
        // Function that computes pressure derivatives at cell centers to apply the fractional step
        int iX_,iXX,iY_,iYY;
        for (int i=0;i<Nq;i++) {
            getAdjacentIndices(i, iX_,iXX,iY_,iYY);
            dpdx(i) = 1/(2*dx) * (u(iXX)-u(iX_));
            dpdy(i) = 1/(2*dy) * (v(iYY)-v(iY_));
        }
    }

    void computeLaplacian() {
        // Function that computes Laplacian on grid faces
        int iX_,iXX,iY_,iYY;
        for (int i=0;i<Nq;i++) {
            getAdjacentIndices(i, iX_,iXX,iY_,iYY);
            lapU(i) = 1/(dx*dx) * (u(iXX)-2*u(i)+u(iX_)) +  1/(dy*dy) * (u(iYY)-2*u(i)+u(iY_));
            lapV(i) = 1/(dx*dx) * (v(iXX)-2*v(i)+v(iX_)) +  1/(dy*dy) * (v(iYY)-2*v(i)+v(iY_));
        }
    }

    void computeConvectiveTerm() {
        interpUVFaceToCorner();
        computeCrossTermAtCorners();
        interpCrossTermCornerToFace();
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
            dvvdy(i) = 1/(2*dy) * (v(iYY)*v(iYY)-v(iY_)*v(iY_));
        }
    }

    void computeCrossTermAtCorners() {
        // Function that computes duvdx and duvdy on grid corners
        int iX_,iXX,iY_,iYY;
        for (int i=0;i<Nq;i++) {
            getAdjacentIndices(i, iX_,iXX,iY_,iYY);
            duvdxC(i) = 1/(2*dx) * (uq(iXX)*vq(iXX)-uq(iX_)*vq(iX_));
            duvdyC(i) = 1/(2*dy) * (uq(iYY)*vq(iYY)-uq(iY_)*vq(iY_));
        }
    }

    // Convenience methods
    void getAdjacentIndices(const int &i, int &iX_,int &iXX,int &iY_,int &iYY) {
        iX_ = i-1;
        iXX = i+1;
        if(i%Nxs==0){iX_=iX_+Nxs;}
        if((i+1)%Nxs==0) {iXX=iXX-Nxs;}
        iY_ = i-Nxs;
        iYY = i+Nxs;
        if(iY_<0) {iY_=iY_+Nq;}
        if(iYY>Nq-1) {iYY=iYY-Nq;}
    }

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

    void interpCrossTermCornerToFace() {
        // Function that linearly interpolates the cross derivative from the lower left corner to the left
        // and bottom face
        // duvdy should be on the left face
        // duvdx should be on the bottom face
        int iX_,iXX,iY_,iYY;
        for (int i=0;i<Nq;i++) {
            getAdjacentIndices(i, iX_,iXX,iY_,iYY);
            duvdx(i) = 0.5*(duvdxC(i) + duvdxC(iXX));
            duvdy(i) = 0.5*(duvdyC(i) + duvdyC(iY_));
        }
    }
};

int main() {
    double Re = 1000;
    double Lx = 1.0;
    double Ly = 1.0;
    double Nx = 64;
    double Ny = 64;
    double T = 0.1;
    double Nt = 1000;
    Simulation sim(Re, Lx, Ly, T, Nx, Ny,Nt);
    for (int i=0;i<Nt;i++) {
        sim.advance();
    }
    return 0;
}


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