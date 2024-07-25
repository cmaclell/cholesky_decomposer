#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

Eigen::MatrixXd choleskyDecomposition(const Eigen::MatrixXd& matrix) {
    // Ensure the matrix is symmetric positive definite
    Eigen::LLT<Eigen::MatrixXd> llt(matrix);
    if(llt.info() == Eigen::NumericalIssue) {
        throw std::runtime_error("Matrix is not positive definite");
    }
    return llt.matrixL(); // Return the lower triangular matrix
}

PYBIND11_MODULE(cholsky_update, m) {
    m.doc() = "Wrapper around eigen's cholsky operations for numpy arrays";
    m.def("add", &add, "A function that adds two numbers");
    m.def("cholesky_decomposition", &choleskyDecomposition, "A function that performs Cholesky decomposition on a given matrix.");

}
