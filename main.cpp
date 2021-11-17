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
    // Spatial discretization
    double Lx,Ly;
    int Nx,Ny;
    int Nxs,Nys,Nq;
    double dx,dy;
    // Temporal discretization
    double T,dt;
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
    MatrixXd duvdxC, duvdyC;
    MatrixXd duudx, dvvdy;
    MatrixXd dpdxm,dpdym,dpdx,dpdy;
    // Time Derivatives
    MatrixXd dudt,dvdt,dudtT,dvdtT;
    // Grid and Poisson setup
    Eigen::MatrixXd gridX,gridY;
    Eigen::MatrixXd kX,kY;
    Eigen::MatrixXd kXmod,kYmod;
    // FFTW setup
    fftw_complex *f, *f_hat,*p, *p_hat;
    fftw_plan forward_plan, inverse_plan;

    // Constructor
    Simulation(double Re_, double Lx_, double Ly_, double T_, int Nx_, int Ny_, int Nt_)
      : Re(Re_), Lx(Lx_),Ly(Ly_),T(T_), Nx(Nx_),Ny(Ny_),Nt(Nt_),dx(Lx_/Nx_),dy(Ly_/Ny_),dt(T_/Nt_),Nxs(Nx_),Nys(Ny_),
        Nq(Nxs*Nys), u(Nq,1), v(Nq,1), uq(Nq,1), vq(Nq,1),lapU(Nq,1), lapV(Nq,1),
        duvdx(Nq,1), duvdy(Nq,1), duvdxC(Nq,1), duvdyC(Nq,1), duudx(Nq,1), dvvdy(Nq,1),
        convU(Nq,1), convV(Nq,1), dudt(Nq,1), dvdt(Nq,1), um(Nq,1), vm(Nq,1), div(Nq,1),omega(Nq,1),
        dpdx(Nq,1), dpdy(Nq,1), dpdxm(Nq,1), dpdym(Nq,1), gridX(Nq,1), gridY(Nq,1), kX(Nq,1), kY(Nq,1), kXmod(Nq,1), kYmod(Nq,1),
        uT(Nq,1), vT(Nq,1),omegaT(Nq,1),dudtT(Nq,1),dvdtT(Nq,1){
        // Compute the grid coordinates and wavenumbers needed for the poisson equation
        computeGridCoordinates();
        writeGridCoordinates();
        // Don't do in place transforms...requires extra considerations
        f = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
        f_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
        p = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
        p_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nq);
        forward_plan = fftw_plan_dft_2d(Nx, Ny, f, f_hat, FFTW_FORWARD, FFTW_MEASURE);
        inverse_plan = fftw_plan_dft_2d(Nx, Ny, p_hat, p, FFTW_BACKWARD, FFTW_MEASURE);
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
        computeRHS();
        k1u = dt * dudt;
        k1v = dt * dvdt;
        u = u + 0.5 * k1u;
        v = v + 0.5 * k1v;
        computeProjectionStep();
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
        // Compute the vorticity for visualization
        computeVorticity();
    }

    void advanceEuler() {
        // Step 1
        computeRHS();
        u = u + dt * dudt;
        v = v + dt * dvdt;
        computeProjectionStep();
        // Compute the vorticity for visualization
        computeVorticity();
    }

    void computeGridCoordinates() {
        // Grid goes from 0 <= x <= 2*pi*Lx-dx, 0 <= y <= Ly-dy
        double y = Ly-dy;
        for (int j=0;j<Ny;j++) {
            double x = 0.0;
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
//                std::cout << "("<<gridX(i+Nx*j)<<","<<gridY(i+Nx*j)<<")"<<std::endl;
            }
            y-=dy;
        }
    }

    void computeRHSPoisson() {
        computeDivergence();
        for (int i=0;i<Nq;i++) {
            f[i][0] = dt*div(i);
            f[i][1] = 0.0;
        }
    }

    void computeProjectionStep() {
        solvePoissonEq();
        computePressureDerivatives();
        interpPressureDerivativeCenterToFace();
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
        p_hat[0][0] = 0.0;
        p_hat[0][1] = 0.0;
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
        u = u - dpdx;
        v = v - dpdy;
    }

    void computeRHS()  {
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
        double meanDiv = 0.0;
        for (int i=0;i<Nq;i++) {
            getAdjacentIndices(i, iX_,iXX,iY_,iYY);
            div(i) = 1/(2*dx) * (um(iXX)-um(iX_)) + 1/(2*dy) * (vm(iYY)-vm(iY_));
            meanDiv += div(i);
        }
        meanDiv/=Nq;
        std::cout<<"Mean divergence is: "<<meanDiv<<std::endl;
    }

    void computeVorticity() {
        interpUVCenter();
        int iX_,iXX,iY_,iYY;
        for (int i=0;i<Nq;i++) {
            getAdjacentIndices(i, iX_,iXX,iY_,iYY);
            omega(i) = 1/(2*dx) * (vm(iXX)-vm(iX_)) - 1/(2*dy) * (um(iYY)-um(iY_));
        }
    }

    void computePressureDerivatives() {
        // Function that computes pressure derivatives at cell centers to apply the fractional step
        int iX_,iXX,iY_,iYY;
        for (int i=0;i<Nq;i++) {
            getAdjacentIndices(i, iX_,iXX,iY_,iYY);
            dpdxm(i) = 1/(2*dx) * (p[iXX][0]-p[iX_][0]);
            dpdym(i) = 1/(2*dy) * (p[iYY][0]-p[iY_][0]);
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

    void interpPressureDerivativeCenterToFace() {
        // Function that linearly interpolates the pressure derivatives from the center to the face
        // the center
        int iX_,iXX,iY_,iYY;
        for (int i=0;i<Nq;i++) {
            getAdjacentIndices(i, iX_,iXX,iY_,iYY);
            dpdx(i) = 0.5 * (dpdxm(i) + dpdxm(iX_));
            dpdy(i) = 0.5 * (dpdym(i) + dpdym(iYY));
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
//            std::cout<<"("<<duvdx(i)<<","<<duvdy(i)<<")"<<std::endl;
        }
    }

    void setTaylorGreenIC(double A, double a, double B, double b) {
        double x,y;
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<double> dist(-0.1,0.1); // distribution in range [-1, 1]
        int count = 0;
        for (int j=0;j<Nx;j++) {
            for (int i=0;i<Ny;i++) {
                x = gridX(count);
                y = gridY(count);
                u(count) = A*sin(a*x)*cos(b*y);
                v(count) = B*cos(a*x)*sin(b*y);
                if (i>Nx/2+7 and i < Nx/2+17 and j>Ny/2+7 and j<Ny/2+17) {
                    u(count) = A*sin(a*x)*cos(b*y) + 1.0;
                    v(count) = B*cos(a*x)*sin(b*y) + 1.0;
//                    u(count) = A*sin(a*x)*cos(b*y) + dist(rng);
//                    v(count) = B*cos(a*x)*sin(b*y) + dist(rng);
                }
                count++;
            }
        }
        computeVorticity();
    }

    void setTestIC() {
        for (int i=0;i<Nq;i++) {
            u(i) = i+1;
            v(i) = i+1;
            duvdxC(i) = i+1;
            duvdyC(i) = i+1;
        }
    }

    void setTestStaggeredGridIC() {
        double x,y;
        for (int i=0;i<Nq;i++) {
            x = gridX(i);
            y = gridY(i);
            u(i) = cos(x);
            v(i) = sin(y);
        }
    }

    void setTestRandomIC() {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<double> dist(-1.0,1.0); // distribution in range [-1, 1]
        double x,y;
        for (int i=0;i<Nq;i++) {
            u(i) = dist(rng);
            v(i) = dist(rng);
        }
    }

    void computeStaggeredGridError() {
        double x,y;
        double error = 0.0;
        for (int i=0;i<Nq;i++) {
            x = gridX(i);
            y = gridY(i);
            dudtT(i) = 2*sin(x)*cos(x) - cos(x)*cos(y)   -cos(x)/Re;
            dvdtT(i) = sin(x)*sin(y)   - 2*cos(y)*sin(y) -sin(y)/Re;
            error = abs(dudtT(i)-dudt(i)) + abs(dvdtT(i)-dvdt(i));
        }
        error/=(2*Nq);
        std::cout<<"error: "<<error<<std::endl;
    }

    double computeTaylorGreenError(double t) {
        double x,y;
        double error = 0.0;
        for (int i=0;i<Nq;i++) {
            x = gridX(i);
            y = gridY(i);
            uT(i) = sin(x)*cos(y)*exp(-2*t/Re);
            vT(i) = -cos(x)*sin(y)*exp(-2*t/Re);
            omegaT(i) = 2*sin(x)*sin(y)*exp(-2*t/Re);
            error = abs(uT(i)-u(i)) + abs(vT(i)-v(i));
        }
        error/=(2*Nq);
        std::cout<<"error: "<<error<<std::endl;
        return error;
    }

    void writeUV(const int &iter) {
        // Writes the solution u,v to a CSV file
        ofstream fu,fv,fomega;
        fu.open("u_" + std::to_string(iter) + ".csv");
        fv.open("v_" + std::to_string(iter) + ".csv");
        fomega.open("omega_" + std::to_string(iter) + ".csv");
        for (int i=0;i<Nq;i++){
            fu << u(i) <<",";
            fv << v(i)<<",";
            fomega << omega(i)<<",";
        }
        fu.close();
        fv.close();
        fomega.close();
    }

    void writeGridCoordinates() {
        // Writes the gridCoordinates
        ofstream fu;
        ofstream fv;
        ofstream fgrid;
        fgrid.open("grid.csv");
        for (int i=0;i<Nq;i++){
            fgrid << gridX(i)<<","<<gridY(i)<<std::endl;
        }
        fgrid.close();
    }
};

int main() {
    double Re = 1000;
    double Lx = 2*M_PI;
    double Ly = 2*M_PI;
    int Nx = 64;
    int Ny = 64;
    double T = 1.0;
    int Nt = 400;

    Simulation sim(Re, Lx, Ly, T, Nx, Ny,Nt);

    // ACTUAL LOOP
    double A,a,B,b;
    A=1.0;a=1.0;B=-1.0;b=1.0;
    sim.setTaylorGreenIC(A,a,B,b);
    sim.writeUV(0);
    sim.computeTaylorGreenError(0.0);

    double t = sim.dt;
    double globalError = 0.0;
    double error = 0.0;
    for (int i=1;i<Nt;i++) {
        std::cout<<"iter="<<i<<std::endl;
        sim.advance();
        sim.writeUV(i);
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

// STAGGERED GRID DIFFERENTIATION IS ORDER 2 ACCURATE
//    sim.setTestStaggeredGridIC();
//    sim.computeRHS();
//    sim.computeStaggeredGridError();

// FRACTIONAL STEP METHOD WORKS -- IF U IS VERY BIG THEN WE GET NANS, I THINK THE PROBLEM HERE IS IN THE NONDIMENSIONALIZATION AND SUBSEQUENT TIME INTEGRATION
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