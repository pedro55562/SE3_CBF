#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>

#include "quadprogpp/src/QuadProg++.hh"
#include "quadprogpp/src/QuadProg++.cc"

#undef solve

#include <Eigen/Core>
#include <Eigen/Dense>
#include <random>
#include <numeric>  // Required for std::iota on windows...

// #include "utils.cuh"
#include "declarations.h"

using namespace std;
using namespace Eigen;

string DistResult::toString() const
{
    std::ostringstream oss;
    oss << "DISTANCE RESULT " << std::endl;
    oss << "D : " << print_number(D) << " m, true_D : " << print_number(true_D) << " m" << std::endl;
    oss << "grad_D : " << print_vector(grad_D) << std::endl;
    return oss.str();
}

PrimInfo::PrimInfo() {};

int mini(int a, int b)
{
    return a < b ? a : b;
}

float maxf(float a, float b)
{
    return (a > b) ? a : b;
}

float minf(float a, float b)
{
    return (a < b) ? a : b;
}

float shape_fun(float x, float cp, float cn, float eps)
{
    float A = (cn + cp) / 2;
    float B = (cp - cn) / 2;
    return eps + A * x - sqrtf(B * B * x * x + eps * eps);
}

float shape_fun_der(float x, float cp, float cn, float eps)
{
    float A = (cn + cp) / 2;
    float B = (cp - cn) / 2;
    return A - (B * B * x) / sqrtf(B * B * x * x + eps * eps);
}

string print_number(float x, int nochar)
{
    float P1 = pow(10, nochar - 3);
    float P2 = 1 / P1;

    float y = P2 * std::round(x * P1);
    string str;
    if (x >= 0)
        str = " " + std::to_string(y).substr(0, nochar - 1);
    else
        str = std::to_string(y).substr(0, nochar - 1);

    while (str.size() < nochar)
        str += "0";

    return str;
}

string print_vector(VectorXf v, int nochar)
{
    string str = "[";
    for (int ind_row = 0; ind_row < v.rows() - 1; ind_row++)
        str += print_number(v[ind_row], nochar) + ", ";

    str += print_number(v[v.rows() - 1], nochar) + "]";

    return str;
}

string print_matrix(MatrixXf M, int nochar)
{
    string str = "";

    for (int ind_row = 0; ind_row < M.rows(); ind_row++)
    {
        str += (ind_row == 0) ? "[" : " ";
        for (int ind_col = 0; ind_col < M.cols(); ind_col++)
            str += print_number(M(ind_row, ind_col), nochar) + "  ";

        str += (ind_row == M.rows() - 1) ? "]" : "\n";
    }

    return str;
}

Matrix3f s_mat(Vector3f v)
{
    Matrix3f res = Matrix3f::Zero();

    res(0, 1) = -v(2);
    res(1, 0) = v(2);
    res(0, 2) = v(1);
    res(2, 0) = -v(1);
    res(1, 2) = -v(0);
    res(2, 1) = v(0);

    return res;
}

Matrix4f rotx(float theta)
{
    float cost = cosf(theta);
    float sint = sinf(theta);

    Matrix4f rot;
    rot << 1, 0, 0, 0,
        0, cost, -sint, 0,
        0, sint, cost, 0,
        0, 0, 0, 1;

    return rot;
}

Matrix4f roty(float theta)
{
    float cost = cosf(theta);
    float sint = sinf(theta);

    Matrix4f rot;
    rot << cost, 0, sint, 0,
        0, 1, 0, 0,
        -sint, 0, cost, 0,
        0, 0, 0, 1;

    return rot;
}

Matrix4f rotz(float theta)
{
    float cost = cosf(theta);
    float sint = sinf(theta);

    Matrix4f rot;
    rot << cost, -sint, 0, 0,
        sint, cost, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    return rot;
}

Matrix4f trn(float x, float y, float z)
{

    Matrix4f trn;
    trn << 1, 0, 0, x,
        0, 1, 0, y,
        0, 0, 1, z,
        0, 0, 0, 1;

    return trn;
}

float urand(float v_min, float v_max)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(v_min, v_max);

    return (float)dist(mt);
}

float urand()
{
    return urand(0.0, 1.0);
}

VectorXf urand_vec(int n, float v_min, float v_max)
{
    VectorXf v(n);

    for (int i = 0; i < n; i++)
        v[i] = urand(v_min, v_max);

    return v;
}

VectorXf urand_vec(VectorXf v_min, VectorXf v_max)
{
    VectorXf v = VectorXf::Zero(v_min.rows());
    for (int i = 0; i < v_min.rows(); i++)
        v[i] = v_min[i] + urand() * (v_max[i] - v_min[i]);

    return v;
}

MatrixXf null_space(MatrixXf A)
{
    FullPivLU<MatrixXf> lu(A);
    MatrixXf Anull = lu.kernel();

    if (Anull.cols() == 1 && Anull.norm() <= 0.001)
        return MatrixXf::Zero(A.cols(), 0);
    else
        return Anull;
}



MatrixXf m_vert_stack(MatrixXf A1, MatrixXf A2)
{
    MatrixXf A(A1.rows() + A2.rows(), A1.cols());
    A << A1, A2;
    return A;
}

MatrixXf m_hor_stack(MatrixXf A1, MatrixXf A2)
{
    return m_vert_stack(A1.transpose(), A2.transpose()).transpose();
}

VectorXf v_ver_stack(VectorXf v1, VectorXf v2)
{
    VectorXf v(v1.rows() + v2.rows());
    v << v1, v2;
    return v;
}

VectorXf v_ver_stack(float v1, VectorXf v2)
{
    VectorXf v(1 + v2.rows());
    v << v1, v2;
    return v;
}

VectorXf v_ver_stack(VectorXf v1, float v2)
{
    VectorXf v(v1.rows() + 1);
    v << v1, v2;
    return v;
}

VectorXf v_ver_stack(float v1, float v2)
{
    VectorXf v(2);
    v << v1, v2;
    return v;
}

vector<float> quadratic_interp(vector<float> x, int N)
{
    int n = x.size();
    MatrixXf A = MatrixXf::Zero(2 * (n - 1), 3 * (n - 1));
    VectorXf b = VectorXf::Zero(3 * (n - 1));

    A(0, 2) = 1;

    int row = 0;

    for (int i = 0; i < n - 1; i++)
    {
        row++;
        A(row, 3 * i) = 1;
        A(row, 3 * i + 1) = 1;
        A(row, 3 * i + 2) = 1;
        b[row] = x[i + 1] - x[i];
        if (i < n - 2)
        {
            row++;
            A(row, 3 * i) = 3;
            A(row, 3 * i + 1) = 2;
            A(row, 3 * i + 2) = 1;
            A(row, 3 * (i + 1) + 2) = -1;
        }
    }

    MatrixXf H = MatrixXf::Zero(3 * (n - 1), 3 * (n - 1));

    for (int i = 0; i < n - 1; i++)
    {
        H(3 * i, 3 * i) = 9.0 / 5.0;
        H(3 * i + 1, 3 * i + 1) = 4.0 / 3.0;
        H(3 * i + 2, 3 * i + 2) = 1.0;

        H(3 * i, 3 * i + 1) = 6.0 / 4.0;
        H(3 * i + 1, 3 * i) = 6.0 / 4.0;

        H(3 * i, 3 * i + 2) = 1.0;
        H(3 * i + 2, 3 * i) = 1.0;

        H(3 * i + 1, 3 * i + 2) = 1.0;
        H(3 * i + 2, 3 * i + 1) = 1.0;
    }

    MatrixXf M = H.inverse() * (A.transpose());

    // Solve the linear system
    VectorXf coef = M * (A * M).inverse() * b;

    vector<float> results;

    float s;
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < N; j++)
        {
            s = ((float)j) / ((float)N);
            results.push_back(coef[3 * i] * s * s * s + coef[3 * i + 1] * s * s + coef[3 * i + 2] * s + x[i]);
        }
    results.push_back(x[n - 1]);

    return results;
}

vector<VectorXf> quadratic_interp(vector<VectorXf> x, int N)
{
    int m = x[0].rows();

    vector<VectorXf> results;
    vector<vector<float>> results_int;

    for (int i = 0; i < m; i++)
    {
        vector<float> xi;
        for (int j = 0; j < x.size(); j++)
            xi.push_back(x[j][i]);

        results_int.push_back(quadratic_interp(xi, N));
    }

    for (int j = 0; j < results_int[0].size(); j++)
    {
        VectorXf xint = VectorXf::Zero(m);
        for (int i = 0; i < m; i++)
            xint[i] = results_int[i][j];

        results.push_back(xint);
    }

    return results;
}

vector<VectorXf> quadratic_interp_vec(vector<VectorXf> x, int N)
{
    return quadratic_interp(x, N);
}

VectorFieldResult vectorfield_rn(VectorXf q, vector<VectorXf> &q_path, float alpha, float const_velocity, bool is_closed, float gamma)
{
    float min_dist = VERYBIGNUMBER;
    float aux_dist;
    int index_q = -1;
    int no_q = q_path.size();
    float EPS = 1.0 / VERYBIGNUMBER;

    int n = q_path[0].rows();

    for (int ind_q = 0; ind_q < no_q; ind_q++)
    {
        aux_dist = (q - q_path[ind_q]).squaredNorm();

        if (aux_dist < min_dist)
        {
            min_dist = aux_dist;
            index_q = ind_q;
        }
    }

    min_dist = sqrtf(min_dist);

    VectorXf N = q_path[index_q] - q;
    N = N / (N.norm() + EPS);

    VectorXf T;

    if (index_q < q_path.size() - 1)
        T = q_path[index_q + 1] - q_path[index_q];
    else
        T = q_path[index_q] - q_path[index_q - 1];

    T = T / (T.norm() + EPS);

    float G = (2 / M_PI) * atan(alpha * sqrtf(min_dist));
    float H = sqrtf(1 + EPS - G * G);

    float mult;

    if(is_closed)
        mult = 1;
    else
    {
        float per = ((float) index_q)/((float) no_q);
        mult = minf(gamma * (1.0-per),1.0);
    }
    

    VectorFieldResult vfr;
    float sign = const_velocity > 0 ? 1 : -1;
    vfr.twist = abs(const_velocity) * (G * N + mult * sign * H * T);
    vfr.dist = min_dist;
    vfr.index = index_q;

    return vfr;
}

PrimDistResult dist_line_aux(Vector3f a0, Vector3f a1, Vector3f b0, Vector3f b1)
{
    MatrixXf H = MatrixXf::Zero(3, 2);
    Vector3f h = b0 - a0;

    H(0, 0) = a1[0] - a0[0];
    H(1, 0) = a1[1] - a0[1];
    H(2, 0) = a1[2] - a0[2];
    H(0, 1) = b0[0] - b1[0];
    H(1, 1) = b0[1] - b1[1];
    H(2, 1) = b0[2] - b1[2];

    Vector2f coef_un = (H.transpose() * H + 0.01 * Matrix2f::Identity()).inverse() * H.transpose() * h;

    float alpha = minf(maxf(coef_un[0], 0.0), 1.0);
    float beta = minf(maxf(coef_un[1], 0.0), 1.0);

    PrimDistResult ldr;

    ldr.proj_A = a0 + (a1 - a0) * alpha;
    ldr.proj_B = b0 + (b1 - b0) * beta;
    ldr.dist = (ldr.proj_A - ldr.proj_B).norm();

    return ldr;
}

PrimDistResult dist_line(vector<Vector3f> a0, vector<Vector3f> a1, vector<Vector3f> b0, vector<Vector3f> b1)
{

    float min_dist = VERYBIGNUMBER;
    Vector3f pointA;
    Vector3f pointB;

    for (int ind_a = 0; ind_a < a0.size(); ind_a++)
        for (int ind_b = 0; ind_b < b0.size(); ind_b++)
        {
            PrimDistResult ldr_temp;
            ldr_temp = dist_line_aux(a0[ind_a], a1[ind_a], b0[ind_b], b1[ind_b]);
            if (ldr_temp.dist < min_dist)
            {
                min_dist = ldr_temp.dist;
                pointA = ldr_temp.proj_A;
                pointB = ldr_temp.proj_B;
            }
        }

    PrimDistResult ldr;

    ldr.dist = min_dist;
    ldr.proj_A = pointA;
    ldr.proj_B = pointB;

    return ldr;
}

VectorXf dp_inv_solve(const MatrixXf& A, const VectorXf& b, float eps) {
    int n = A.rows();
    int m = A.cols();
    
    MatrixXf M(n + m, n + m);
    VectorXf rhs(n + m);
    
    M.topLeftCorner(m, m) = eps * MatrixXf::Identity(m, m);
    M.topRightCorner(m, n) = -A.transpose();
    M.bottomLeftCorner(n, m) = A;
    M.bottomRightCorner(n, n) = MatrixXf::Identity(n, n);
    
    rhs.head(m).setZero();
    rhs.tail(n) = b;
    
    VectorXf solution = M.colPivHouseholderQr().solve(rhs);
    
    return solution.head(m);
}

VectorXf sqrt_sign(VectorXf v)
{
    VectorXf w = VectorXf(v.rows());

    for(int i=0; i < v.rows(); i++)
        w[i] = v[i] > 0? sqrtf(v[i]): -sqrtf(-v[i]);

    return w;
}

vector<Vector3f> get_vertex(const MatrixXf& A, const VectorXf& b) {
    vector<Vector3f> vertices;
    int num_constraints = A.rows();
    int dim = A.cols();

    if (num_constraints < 3 || dim != 3) {
        cerr << "Error: A must have at least 3 constraints and be in 3D." << endl;
        return vertices;
    }

    vector<int> indices(num_constraints);
    iota(indices.begin(), indices.end(), 0); 

    for (size_t i = 0; i < num_constraints; ++i) {
        for (size_t j = i + 1; j < num_constraints; ++j) {
            for (size_t k = j + 1; k < num_constraints; ++k) {
                MatrixXf A_sub(3, 3);
                VectorXf b_sub(3);

                A_sub.row(0) = A.row(i);
                A_sub.row(1) = A.row(j);
                A_sub.row(2) = A.row(k);

                b_sub(0) = b(i);
                b_sub(1) = b(j);
                b_sub(2) = b(k);

                FullPivLU<MatrixXf> lu(A_sub);
                if (!lu.isInvertible()) continue; 

                Vector3f p = lu.solve(b_sub);

                VectorXf check = A * p - b;

                
                if (check.maxCoeff()<=1e-5) {
                    vertices.push_back(p);
                }
            }
        }
    }

    return vertices;
}

VectorXf solveQP(const MatrixXf& H,const VectorXf& f,const MatrixXf& A,const VectorXf& b,const MatrixXf& Aeq,const VectorXf& beq) 
{
    // Solve min_u (u'*H*u)/2 + f'*u
    // such that:
    // A*u >= b
    // Aeq*u = beq
    // The function assumes that H is a positive definite function (the problem is strictly convex)

    int n = H.rows();
    int meq = Aeq.rows();
    int mineq = A.rows();


    quadprogpp::Matrix<double> H_aux, Aeq_aux, A_aux;
    quadprogpp::Vector<double> f_aux, beq_aux, b_aux, u_aux;

    H_aux.resize(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            H_aux[i][j] = H(i, j);

    f_aux.resize(n);
    for (int i = 0; i < n; i++)
        f_aux[i] = f[i];

    Aeq_aux.resize(n, meq);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < meq; j++)
            Aeq_aux[i][j] = Aeq(j, i);

    beq_aux.resize(meq);
    for (int j = 0; j < meq; j++)
        beq_aux[j] = -beq[j];

    A_aux.resize(n, mineq);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < mineq; j++)
            A_aux[i][j] = A(j, i);

    b_aux.resize(mineq);
    for (int j = 0; j < mineq; j++)
        b_aux[j] = -b[j];

    u_aux.resize(n);

    double val = solve_quadprog(H_aux, f_aux, Aeq_aux, beq_aux, A_aux, b_aux, u_aux);

    if (val > 1.0E50)
    {
        // Problem is unfeasible
        VectorXf u(0);
        return u;
    }
    else
    {
        // Problem is feasible
        VectorXf u(n);

        for (int i = 0; i < n; i++)
            u[i] = u_aux[i];

        return u;
    }
}

VectorXf solveQP(const MatrixXf& H,const VectorXf& f,const MatrixXf& A,const VectorXf& b) 
{
    int n = H.rows();
    return solveQP(H,f,A,b,MatrixXf::Zero(0, n),VectorXf::Zero(0));
}

