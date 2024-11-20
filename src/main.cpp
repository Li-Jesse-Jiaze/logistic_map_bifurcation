#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <unsupported/Eigen/MPRealSupport>
#include <unsupported/Eigen/SparseExtra>
#include <mpfr.h>
#include <mpreal.h>
#include <omp.h>

#include "tictoc.h"

#define PRECISION 400

// (Saha and Strogatz 1995)
void computeFunction(int order, const Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1> &x, Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1> &function_values)
{
    mpfr::mpreal mu = x(order - 1);

    #pragma omp parallel for
    for (int i = 0; i < (order - 2); i++)
    {
        function_values(i) = mu * x(i) * (1.0 - x(i)) - x(i + 1);
    }
    function_values(order - 2) = mu * x(order - 2) * (1.0 - x(order - 2)) - x(0);

    mpfr::mpreal product = 1.0;
    for (int i = 0; i < (order - 1); i++)
    {
        product *= mu * (1 - 2 * x(i));
    }
    function_values(order - 1) = product + 1;
}

void initializeJacobian(int order, Eigen::SparseMatrix<mpfr::mpreal> &jacobian)
{
    std::vector<Eigen::Triplet<mpfr::mpreal>> tripletList;
    tripletList.reserve(3 * (order - 1) + 1);

    for (int i = 0; i < (order - 2); i++)
    {
        tripletList.emplace_back(i, i, 1.0);
        tripletList.emplace_back(i, i + 1, -1.0);
        tripletList.emplace_back(i, order - 1, 1.0);
    }
    tripletList.emplace_back(order - 2, order - 2, 1.0);
    tripletList.emplace_back(order - 2, 0, -1.0);
    tripletList.emplace_back(order - 2, order - 1, 1.0);

    for (int k = 0; k < (order - 1); k++)
    {
        tripletList.emplace_back(order - 1, k, 1.0);
    }
    tripletList.emplace_back(order - 1, order - 1, 1.0);

    jacobian.setFromTriplets(tripletList.begin(), tripletList.end());
}

void computeJacobian(int order, const Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1> &x, Eigen::SparseMatrix<mpfr::mpreal> &jacobian)
{
    const mpfr::mpreal mu = x(order - 1);

    // intermediate = 1 - 2 * x(i)
    Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1> intermediate(order - 1);
    #pragma omp parallel for
    for(int i = 0; i < order - 1; ++i) {
        intermediate(i) = 1 - 2 * x(i);
    }

    // mu * intermediate(i)
    Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1> mu_intermediate(order - 1);
    #pragma omp parallel for
    for(int i = 0; i < order - 1; ++i) {
        mu_intermediate(i) = mu * intermediate(i);
    }

    mpfr::mpreal all_product = 1.0;
    for(int i = 0; i < order - 1; ++i) {
        all_product *= mu_intermediate(i);
    }

    std::vector<Eigen::Triplet<mpfr::mpreal>> tripletList;
    tripletList.reserve(3 * (order - 1) + (order - 1) + 1);

    // first 3*(order-1) items
    for(int i = 0; i < order - 2; ++i) {
        tripletList.emplace_back(i, i, mu_intermediate(i));
        tripletList.emplace_back(i, i + 1, -1.0);
        tripletList.emplace_back(i, order - 1, x(i) * (1 - x(i)));
    }
    // last row
    int last = order - 2;
    tripletList.emplace_back(last, last, mu_intermediate(last));
    tripletList.emplace_back(last, 0, -1.0);
    tripletList.emplace_back(last, order - 1, x(last) * (1 - x(last)));

    tripletList.reserve(tripletList.size() + order - 1);
    #pragma omp parallel
    {
        std::vector<Eigen::Triplet<mpfr::mpreal>> local_triplets;
        #pragma omp for nowait
        for(int k = 0; k < order - 1; ++k) {
            // temp = all_product / (mu * intermediate(k))
            mpfr::mpreal temp = all_product / mu_intermediate(k);
            local_triplets.emplace_back(order - 1, k, -2.0 * temp * mu);
        }

        #pragma omp critical
        {
            tripletList.insert(tripletList.end(), local_triplets.begin(), local_triplets.end());
        }
    }

    // last col
    mpfr::mpreal final_temp = all_product;
    tripletList.emplace_back(order - 1, order - 1, (order - 1) * final_temp / mu);

    jacobian.setZero();
    jacobian.setFromTriplets(tripletList.begin(), tripletList.end());
}

void bifurcationCalculation(int order, mpfr::mpreal &r_order, mpfr::mpreal &initial_value, const mpfr::mpreal &tolerance)
{
    mpfr::mpreal::set_default_prec(PRECISION);

    Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1> function_values(order, 1);
    Eigen::SparseMatrix<mpfr::mpreal> jacobian(order, order);
    Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1> newton_step(order, 1);
    Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1> x_values(order, 1);
    Eigen::SparseLU<Eigen::SparseMatrix<mpfr::mpreal>, Eigen::COLAMDOrdering<int>> solver;

    initializeJacobian(order, jacobian);

    // init vector
    mpfr::mpreal exponent = 3.3 * log10(mpfr::mpreal(order - 1));
    mpfr::mpreal base = 0.21;
    mpfr::mpreal delta_lambda = pow(base, exponent);
    r_order += delta_lambda;

    for(int i = 0; i < 10 * order; i++) {
        initial_value = r_order * initial_value * (1 - initial_value);
    }

    // fill x_values
    for(int i = 0; i < order - 1; i++) {
        initial_value = r_order * initial_value * (1 - initial_value);
        x_values(i) = initial_value;
    }
    x_values(order - 1) = r_order;

    // Newton
    for(int iteration = 0; iteration < 75; iteration++) {
        // TicToc t_f;
        computeFunction(order, x_values, function_values);
        // printf("computeFunction: %f ms\n", t_f.toc());

        // TicToc t_j;
        computeJacobian(order, x_values, jacobian);
        // printf("computeJacobian: %f ms\n", t_j.toc());

        if(iteration == 0) {
            solver.analyzePattern(jacobian);
        }

        solver.factorize(jacobian);
        newton_step = solver.solve(function_values);
        if(solver.info() != Eigen::Success) {
            throw std::runtime_error("Solving failed");
        }

        // update x_values
        for(int i = 0; i < order; i++) {
            x_values(i) -= newton_step(i);
        }

        if(newton_step.norm() < tolerance) {
            r_order = x_values(order - 1);
            initial_value = x_values(0);
            return;
        }
    }

    throw std::runtime_error("Convergence failed!");
}

int main(int argc, char *argv[])
{
    mpfr::mpreal::set_default_prec(PRECISION);
    mpfr::mpreal initial_value = 8.0e-01;
    mpfr::mpreal r_order = 3.44;
    int order = 4;

    std::string filename = "results_bifurcation.txt";
    std::ofstream outfile(filename.c_str(), std::ofstream::out);

    outfile.precision(PRECISION / 3);
    outfile << "PRECISION = " << PRECISION << ", tolerance = " << 1e-100 << "\n";

    for(int i = 0; i < 21; i++) {
        std::cout << "order = " << order << std::endl;

        try {
            TicToc t_b;
            bifurcationCalculation(order + 1, r_order, initial_value, 1e-100);
            printf("bifurcationCalculation: %f ms\n", t_b.toc());
            std::cout << "r_order = " << r_order << std::endl;
            outfile << "order = " << order << ", r_order = " << r_order << std::endl;
            outfile.flush();
        }
        catch(const std::exception &e) {
            std::cerr << e.what() << std::endl;
            break;
        }

        order *= 2;
    }

    outfile.close();
}