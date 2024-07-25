#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;

class CholeskyDecomposer {
public:
    // Initialize with an empty constructor
    CholeskyDecomposer() {}

    // Initialize the Cholesky decomposition with a given matrix
    void compute(const Eigen::MatrixXd& matrix) {
        llt.compute(matrix);
        if (llt.info() == Eigen::NumericalIssue) {
            throw std::runtime_error("Matrix is not positive definite");
        }
    }

    // Perform a rank-1 update
    void rankUpdate(const Eigen::VectorXd& vec, double alpha) {
        llt.rankUpdate(vec, alpha);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("Rank update failed");
        }
    }

    // Retrieve the current lower triangular matrix
    Eigen::MatrixXd matrixL() const {
        return llt.matrixL();
    }

private:
    Eigen::LLT<Eigen::MatrixXd> llt;
};

PYBIND11_MODULE(py_eigen, m) {
    m.doc() = "A light weight wrapper around the C++ eigen's library for efficient operations on numpy arrays";

    py::class_<CholeskyDecomposer>(m, "CholeskyDecomposer")
        .def(py::init<>())
        .def("decompose", &CholeskyDecomposer::compute, "Initialize the Cholesky decomposition with a given matrix.")
        .def("rank_update", &CholeskyDecomposer::rankUpdate, "Perform a rank-1 update on the Cholesky decomposition.")
        .def("get_L", &CholeskyDecomposer::matrixL, "Retrieve the current lower triangular matrix.");
}

