#include <fstream>
#include <iostream>
#include <sstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <thread>
#include <mutex>
#include "gjk.h"
#include "nanoflann.hpp"
#include <queue>
#include <chrono>
#include <random>


#include <condition_variable>
#include <functional>
#include <atomic>


#include "declarations.h"

#include <limits>
#include <stdexcept>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace Eigen;
using namespace std::chrono;

////////////////////////////////////////////////////////////////
// FORWARD KINEMATIC
////////////////////////////////////////////////////////////////

FKResult::FKResult(int _no_links) : no_links(_no_links)
{
    htm_dh = vector<Matrix4f>(no_links);
    jac_v_dh = vector<MatrixXf>(no_links);
    jac_w_dh = vector<MatrixXf>(no_links);
};

Vector3f FKResult::get_x_dh(int ind_link) const { return htm_dh[ind_link].block<3, 1>(0, 0); }

Vector3f FKResult::get_y_dh(int ind_link) const { return htm_dh[ind_link].block<3, 1>(0, 1); }

Vector3f FKResult::get_z_dh(int ind_link) const { return htm_dh[ind_link].block<3, 1>(0, 2); }

Vector3f FKResult::get_p_dh(int ind_link) const { return htm_dh[ind_link].block<3, 1>(0, 3); }

Matrix3f FKResult::get_Q_dh(int ind_link) const { return htm_dh[ind_link].block<3, 3>(0, 0); }

Vector3f FKResult::get_x_ee() const { return htm_ee.block<3, 1>(0, 0); }

Vector3f FKResult::get_y_ee() const { return htm_ee.block<3, 1>(0, 1); }

Vector3f FKResult::get_z_ee() const { return htm_ee.block<3, 1>(0, 2); }

Vector3f FKResult::get_p_ee() const { return htm_ee.block<3, 1>(0, 3); }

Matrix3f FKResult::get_Q_ee() const { return htm_ee.block<3, 3>(0, 0); }

FKResult::FKResult() {};

string FKResult::toString() const
{
    std::ostringstream oss;
    oss << "FORWARD KINEMATIC " << std::endl;
    oss << "End effector htm: " << std::endl;
    oss << print_matrix(htm_ee) << std::endl;
    oss << "End effector Jacobian (linear): " << std::endl;
    oss << print_matrix(jac_v_ee) << std::endl;
    oss << "End effector Jacobian (angular): " << std::endl;
    oss << print_matrix(jac_w_ee) << std::endl;

    return oss.str();
}

////////////////////////////////////////////////////////////////
// FORWARD KINEMATIC FOR PRIMITIVE
////////////////////////////////////////////////////////////////

FKPrimResult::FKPrimResult() {};

string FKPrimResult::toString() const
{
    std::ostringstream oss;
    oss << "FORWARD KINEMATIC FOR PRIMITIVES" << std::endl;
    return oss.str();
}

////////////////////////////////////////////////////////////////
// INVERSE KINEMATIC
////////////////////////////////////////////////////////////////

IKResult::IKResult() {};

string IKResult::toString() const
{
    std::ostringstream oss;
    oss << "INVERSE KINEMATIC RESULT" << std::endl;
    oss << "qf =" << print_vector(qf) << std::endl;
    oss << "Position error: " << print_number(error_pos) << " m" << std::endl;
    oss << "Orientation error: " << print_number(error_ori) << " deg" << std::endl;
    oss << "Success: " << success << std::endl;
    return oss.str();
}

////////////////////////////////////////////////////////////////
// TASK KINEMATIC
////////////////////////////////////////////////////////////////

TaskResult::TaskResult() {};

string TaskResult::toString() const
{
    std::ostringstream oss;
    oss << "TASK RESULT" << std::endl;
    oss << "Maximum position error: " << print_number(max_error_pos) << " m " << std::endl;
    oss << "Maximum orientation error: " << print_number(max_error_ori) << " deg " << std::endl;

    return oss.str();
}

string remove_space(const string &input)
{
    string result;
    for (char ch : input)
        if (ch != ' ')
            result += ch;

    return result;
}

int extract_key(const string &str, const string &key_str)
{
    string key = "\"" + key_str + "\": ";
    auto start_pos = str.find(key);
    start_pos += key.length();
    auto end_pos = str.find(',', start_pos);
    if (end_pos == string::npos)
        end_pos = str.length();

    string number_str = str.substr(start_pos, end_pos - start_pos);
    return std::stoi(number_str);
}

float extract_no(const string &str)
{
    std::istringstream iss(str);
    float number = 0.0;
    iss >> number;
    return number;
}

////////////////////////////////////////////////////////////////
// GEOMETRIC PRIMITIVES
////////////////////////////////////////////////////////////////

GeometricPrimitives::GeometricPrimitives() {};

string GeometricPrimitives::toString() const
{
    std::ostringstream oss;
    oss << "GEOMETRIC PRIMITIVE" << std::endl;
    if (type == 0)
        oss << "Sphere with radius " << print_number(lx) << " m" << std::endl;
    if (type == 1)
        oss << "Box with width " << print_number(lx) << " m, depth " << print_number(ly) << ", height " << print_number(lz) << " m" << std::endl;
    if (type == 2)
        oss << "Cylinder with radius " << print_number(lx) << " m,  height " << print_number(lz) << " m" << std::endl;
    if (type == 3)
        oss << "Point cloud with " << points_gp.size() << " points " << std::endl;
    if (type == 4)
        oss << "Polytope with " << points_gp.size() << " vertexes " << std::endl;

    return oss.str();
}

GeometricPrimitives GeometricPrimitives::create_sphere(Matrix4f htm, float radius)
{
    GeometricPrimitives gp = GeometricPrimitives();
    gp.lx = radius;
    gp.ly = radius;
    gp.lz = radius;
    gp.htm = htm;
    gp.type = 0;

    return gp;
}

GeometricPrimitives GeometricPrimitives::create_box(Matrix4f htm, float width, float depth, float height)
{
    GeometricPrimitives gp = GeometricPrimitives();
    gp.lx = width;
    gp.ly = depth;
    gp.lz = height;
    gp.htm = htm;
    gp.type = 1;

    return gp;
}

GeometricPrimitives GeometricPrimitives::create_cylinder(Matrix4f htm, float radius, float height)
{
    GeometricPrimitives gp = GeometricPrimitives();
    gp.lx = radius;
    gp.ly = radius;
    gp.lz = height;
    gp.htm = htm;
    gp.type = 2;

    return gp;
}

GeometricPrimitives GeometricPrimitives::create_pointcloud(vector<Vector3f> &points)
{
    GeometricPrimitives gp = GeometricPrimitives();

    gp.pointcloud = std::make_shared<nanoflann::PointCloud<float>>();
    gp.pointcloud->pts.resize(points.size());

    gp.points_gp = {};

    float lx_min = 1e6;
    float lx_max = -1e6;
    float ly_min = 1e6;
    float ly_max = -1e6;
    float lz_min = 1e6;
    float lz_max = -1e6;

    for (int i = 0; i < points.size(); i++)
    {
        gp.pointcloud->pts[i].x = points[i][0];
        gp.pointcloud->pts[i].y = points[i][1];
        gp.pointcloud->pts[i].z = points[i][2];

        lx_min = min(lx_min, gp.pointcloud->pts[i].x);
        ly_min = min(ly_min, gp.pointcloud->pts[i].y);
        lz_min = min(lz_min, gp.pointcloud->pts[i].z);
        lx_max = max(lx_max, gp.pointcloud->pts[i].x);
        ly_max = max(ly_max, gp.pointcloud->pts[i].y);
        lz_max = max(lz_max, gp.pointcloud->pts[i].z);
        gp.points_gp.push_back(points[i]);
    }

    gp.lx = lx_max - lx_min;
    gp.ly = ly_max - ly_min;
    gp.lz = lz_max - lz_min;
    gp.htm = trn((lx_max + lx_min) / 2, (ly_max + ly_min) / 2, (lz_max + lz_min) / 2);
    gp.center = Vector3f((lx_max + lx_min) / 2, (ly_max + ly_min) / 2, (lz_max + lz_min) / 2);

    gp.kdtree = std::make_shared<nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, nanoflann::PointCloud<float>>,
        nanoflann::PointCloud<float>, 3>>(3, *gp.pointcloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

    gp.type = 3;

    gp.bvh = BVH(points);

    return gp;
}

GeometricPrimitives GeometricPrimitives::create_convexpolytope(Matrix4f htm, MatrixXf A, VectorXf b)
{
    GeometricPrimitives gp = GeometricPrimitives();
    // Check if the polytope is empty
    Matrix3f Q = htm.block<3, 3>(0, 0);
    Vector3f p = htm.block<3, 1>(0, 3);

    MatrixXf A_mod = A * Q;
    VectorXf b_mod = b - A * p;

    gp.points_gp = get_vertex(A_mod, b_mod);

    if (gp.points_gp.size() == 0)
        throw std::runtime_error("Polytope is empty!");

    // Check if the polytope is unbounded
    VectorXf ex_p = solveQP(Matrix3f::Identity() / VERYBIGNUMBER, Vector3f(1, 0, 0), -A_mod, -b_mod);
    VectorXf ex_n = solveQP(Matrix3f::Identity() / VERYBIGNUMBER, Vector3f(-1, 0, 0), -A_mod, -b_mod);
    VectorXf ey_p = solveQP(Matrix3f::Identity() / VERYBIGNUMBER, Vector3f(0, 1, 0), -A_mod, -b_mod);
    VectorXf ey_n = solveQP(Matrix3f::Identity() / VERYBIGNUMBER, Vector3f(0, -1, 0), -A_mod, -b_mod);
    VectorXf ez_p = solveQP(Matrix3f::Identity() / VERYBIGNUMBER, Vector3f(0, 0, 1), -A_mod, -b_mod);
    VectorXf ez_n = solveQP(Matrix3f::Identity() / VERYBIGNUMBER, Vector3f(0, 0, -1), -A_mod, -b_mod);

    float dx = abs(ex_p[0] - ex_n[0]);
    float dy = abs(ey_p[1] - ey_n[1]);
    float dz = abs(ez_p[2] - ez_n[2]);

    if (dx > 1e3 || dy > 1e3 || dz > 1e3)
        throw std::runtime_error("Polytope is unbounded!");

    int num_constraints = A.rows();
    int dim = A.cols();

    MatrixXf A_norm = MatrixXf(num_constraints, dim);
    VectorXf b_norm = VectorXf(num_constraints);

    for (int i = 0; i < num_constraints; ++i)
    {
        float row_norm = A_mod.row(i).norm();

        if (row_norm > 1e-6)
        {
            A_norm.row(i) = A_mod.row(i) / row_norm;
            b_norm(i) = b_mod(i) / row_norm;
        }
        else
        {
            A_norm.row(i) = A_mod.row(i);
            b_norm(i) = b_mod(i);
        }
    }

    gp.type = 4;
    gp.htm = htm;
    gp.A = A_norm;
    gp.b = b_norm;

    float x_min = VERYBIGNUMBER;
    float x_max = -VERYBIGNUMBER;
    float y_min = VERYBIGNUMBER;
    float y_max = -VERYBIGNUMBER;
    float z_min = VERYBIGNUMBER;
    float z_max = -VERYBIGNUMBER;

    Vector3f tr_point;

    for (int i = 0; i < gp.points_gp.size(); i++)
    {
        tr_point = gp.points_gp[i];
        x_min = minf(x_min, tr_point[0]);
        x_max = maxf(x_max, tr_point[0]);
        y_min = minf(y_min, tr_point[1]);
        y_max = maxf(y_max, tr_point[1]);
        z_min = minf(z_min, tr_point[2]);
        z_max = maxf(z_max, tr_point[2]);
    }
    gp.center = Vector3f((x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2);
    gp.lx = x_max - x_min;
    gp.ly = y_max - y_min;
    gp.lz = z_max - z_min;
    gp.htm = htm;

    return gp;
}

GeometricPrimitives generate_point_cloud_sphere(float radius, Matrix4f htm, float delta)
{

    vector<Vector3f> points;
    Vector3f ptemp;
    Vector3f p = htm.block<3, 1>(0, 3);

    int N = ceil(radius / delta) + 2;
    int M;

    double x, y, z, phi, theta;

    for (int n = 0; n < N; n++)
    {
        phi = M_PI * ((float)n) / ((float)N - 1);
        M = (int)ceil(N * sin(phi)) + 2;
        for (int m = 0; m < M; m++)
        {
            theta = 2 * M_PI * (((float)m) / ((float)M - 1));

            x = radius * sin(phi) * cos(theta);
            y = radius * sin(phi) * sin(theta);
            z = radius * cos(phi);
            ptemp << x, y, z;
            points.push_back(ptemp + p);
        }
    }

    return GeometricPrimitives::create_pointcloud(points);
}

GeometricPrimitives generate_point_cloud_box(float width, float depth, float height, Matrix4f htm, float delta)
{

    vector<Vector3f> points;
    Vector3f ptemp;
    Matrix3f Q = htm.block<3, 3>(0, 0);
    Vector3f p = htm.block<3, 1>(0, 3);

    int W = ceil(width / delta) + 2;
    int D = ceil(depth / delta) + 2;
    int H = ceil(height / delta) + 2;
    float x, y, z;

    x = -width / 2;
    for (int d = 0; d < D; d++)
    {
        y = -depth / 2 + (depth / (D - 1)) * d;
        for (int h = 0; h < H; h++)
        {
            z = -height / 2 + (height / (H - 1)) * h;
            ptemp << x, y, z;
            points.push_back(Q * ptemp + p);
        }
    }

    x = width / 2;
    for (int d = 0; d < D; d++)
    {
        y = -depth / 2 + (depth / (D - 1)) * d;
        for (int h = 0; h < H; h++)
        {
            z = -height / 2 + (height / (H - 1)) * h;
            ptemp << x, y, z;
            points.push_back(Q * ptemp + p);
        }
    }

    y = -depth / 2;
    for (int w = 0; w < W; w++)
    {
        x = -width / 2 + (width / (W - 1)) * w;
        for (int h = 0; h < H; h++)
        {
            z = -height / 2 + (height / (H - 1)) * h;
            ptemp << x, y, z;
            points.push_back(Q * ptemp + p);
        }
    }

    y = depth / 2;
    for (int w = 0; w < W; w++)
    {
        x = -width / 2 + (width / (W - 1)) * w;
        for (int h = 0; h < H; h++)
        {
            z = -height / 2 + (height / (H - 1)) * h;
            ptemp << x, y, z;
            points.push_back(Q * ptemp + p);
        }
    }

    z = -height / 2;
    for (int w = 0; w < W; w++)
    {
        x = -width / 2 + (width / (W - 1)) * w;
        for (int d = 0; d < D; d++)
        {
            y = -depth / 2 + (depth / (D - 1)) * d;
            ptemp << x, y, z;
            points.push_back(Q * ptemp + p);
        }
    }

    z = height / 2;
    for (int w = 0; w < W; w++)
    {
        x = -width / 2 + (width / (W - 1)) * w;
        for (int d = 0; d < D; d++)
        {
            y = -depth / 2 + (depth / (D - 1)) * d;
            ptemp << x, y, z;
            points.push_back(Q * ptemp + p);
        }
    }

    return GeometricPrimitives::create_pointcloud(points);
}

GeometricPrimitives generate_point_cloud_cylinder(float radius, float height, Matrix4f htm, float delta)
{

    vector<Vector3f> points;
    Vector3f ptemp;
    Matrix3f Q = htm.block<3, 3>(0, 0);
    Vector3f p = htm.block<3, 1>(0, 3);

    int T = ceil(2 * M_PI * radius / delta) + 2;
    int R = ceil(radius / delta) + 2;
    int H = ceil(height / delta) + 2;
    float x, y, z, v, u;

    for (int t = 0; t < T; t++)
    {
        u = 2 * M_PI * ((float)t) / ((float)T - 1);
        for (int h = 0; h < H; h++)
        {
            v = ((float)h) / ((float)H - 1);

            x = radius * cos(u);
            y = radius * sin(u);
            z = -height / 2 + v * (height);
            ptemp << x, y, z;
            points.push_back(Q * ptemp + p);
        }
    }

    for (int r = 0; r < R; r++)
    {
        v = radius * (((float)r) / ((float)R - 1));
        T = ceil(2 * M_PI * v / delta) + 2;
        for (int t = 0; t < T; t++)
        {
            u = 2 * M_PI * ((float)t) / ((float)T - 1);

            x = v * cos(u);
            y = v * sin(u);

            z = -height / 2;
            ptemp << x, y, z;
            points.push_back(Q * ptemp + p);

            z = height / 2;
            ptemp << x, y, z;
            points.push_back(Q * ptemp + p);
        }
    }

    return GeometricPrimitives::create_pointcloud(points);
}

GeometricPrimitives generate_point_cloud_convexpolygon(const vector<Vector3f> &vertices, const MatrixXf &A, const VectorXf &b, const MatrixXf &htm, float disc)
{
    vector<Vector3f> all_face_points;
    const float eps = 1e-6f;

    Matrix3f Q = htm.block<3, 3>(0, 0);
    Vector3f p = htm.block<3, 1>(0, 3);

    for (int i = 0; i < A.rows(); i++)
    {
        Vector3f a = A.row(i);
        float bi = b(i);

        vector<Vector3f> face_vertices;
        for (const auto &v : vertices)
        {
            if (fabs(a.dot(v) - bi) < eps)
                face_vertices.push_back(v);
        }

        if (face_vertices.empty())
            continue;

        Vector3f n = a.normalized();
        Vector3f arbitrary = (fabs(n.x()) < 0.9f) ? Vector3f(1, 0, 0) : Vector3f(0, 1, 0);
        Vector3f d1 = (arbitrary - arbitrary.dot(n) * n).normalized();
        Vector3f d2 = n.cross(d1);

        Vector3f origin = face_vertices[0];
        float u_min = numeric_limits<float>::max(), u_max = numeric_limits<float>::lowest();
        float v_min = numeric_limits<float>::max(), v_max = numeric_limits<float>::lowest();

        for (const auto &v : face_vertices)
        {
            Vector3f diff = v - origin;
            float u = diff.dot(d1);
            float v_coord = diff.dot(d2);
            if (u < u_min)
                u_min = u;
            if (u > u_max)
                u_max = u;
            if (v_coord < v_min)
                v_min = v_coord;
            if (v_coord > v_max)
                v_max = v_coord;
        }

        for (float u = u_min; u <= u_max + eps; u += disc)
        {
            for (float v_coord = v_min; v_coord <= v_max + eps; v_coord += disc)
            {

                Vector3f candidate = origin + u * d1 + v_coord * d2;
                bool inside = true;
                for (int j = 0; j < A.rows(); j++)
                {
                    if (A.row(j).dot(candidate) > b(j) + eps)
                    {
                        inside = false;
                        break;
                    }
                }
                if (inside)
                {
                    all_face_points.push_back(candidate);
                }
            }
        }
    }

    vector<Vector3f> all_face_points_tr;

    for (int i = 0; i < all_face_points.size(); i++)
        all_face_points_tr.push_back(Q * all_face_points[i] + p);

    return GeometricPrimitives::create_pointcloud(all_face_points_tr);
}

inline double smf(double x, int order, double h)
{
    if (x < 0)
    {
        return 0.0;
    }

    if (h == 0)
    {
        switch (order)
        {
        case 2:
            return 1;
        case 1:
            return x;
        case 0:
            return 0.5 * x * x;
        default:
            throw std::invalid_argument("Invalid order. Must be 0, 1, or 2.");
        }
    }
    else
    {
        switch (order)
        {
        case 2:
            return 1.0 - std::pow(x + 1, -1.0 / h);
        case 1:
            return x - (h / (h - 1)) * (std::pow(x + 1, 1.0 - 1.0 / h) - 1);
        case 0:
            return 0.5 * x * x - (h / (h - 1)) * ((h / (2 * h - 1)) * (std::pow(x + 1, 2.0 - 1.0 / h) - 1) - x);
        default:
            throw std::invalid_argument("Invalid order. Must be 0, 1, or 2.");
        }
    }
}

ProjResult projection_sphere(float radius, Matrix4f htm, Vector3f point, float h, float eps)
{
    Vector3f pc = htm.block<3, 1>(0, 3);
    Vector3f deltap = point - pc;
    float radius_xyz = deltap.norm();
    float delta_radius = radius_xyz - radius;

    float e_R = smf(delta_radius, 0, h);
    float e_dR = smf(delta_radius, 1, h);

    ProjResult pr;

    pr.dist = sqrtf(2 * e_R);
    pr.proj = point - e_dR * deltap / radius_xyz;

    return pr;
}

inline Matrix3f projection_sphere_jacobian(float radius, Matrix4f htm,
                                           Vector3f point, float h, float eps)
{
    Vector3f pc = htm.block<3, 1>(0, 3);
    Vector3f d = point - pc;

    float r = d.norm();
    Matrix3f I = Matrix3f::Identity();


    Vector3f u = d / r;

    float delta_radius = r - radius;

    float e_dR  = smf(delta_radius, 1, h);
    float e_ddR = smf(delta_radius, 2, h);

    Matrix3f Hess =
        e_ddR * (u * u.transpose())
        + (e_dR / r) * (I - u * u.transpose());

    return I - Hess;
}



ProjResult projection_box(float lx, float ly, float lz, Matrix4f htm, Vector3f point, float h, float eps)
{
    Vector3f pc = htm.block<3, 1>(0, 3);
    Matrix3f Q = htm.block<3, 3>(0, 0);
    Vector3f point_transformed = Q.transpose() * (point - pc);
    float x = point_transformed[0];
    float y = point_transformed[1];
    float z = point_transformed[2];

    float G_x = (smf(x - lx / 2, 0, h) + smf(-x - lx / 2, 0, h));
    float G_y = (smf(y - ly / 2, 0, h) + smf(-y - ly / 2, 0, h));
    float G_z = (smf(z - lz / 2, 0, h) + smf(-z - lz / 2, 0, h));

    float G = G_x + G_y + G_z;

    float dG_x = (smf(x - lx / 2, 1, h) - smf(-x - lx / 2, 1, h));
    float dG_y = (smf(y - ly / 2, 1, h) - smf(-y - ly / 2, 1, h));
    float dG_z = (smf(z - lz / 2, 1, h) - smf(-z - lz / 2, 1, h));

    float cr = 1.2 * (lx * lx / 4 + ly * ly / 4 + lz * lz / 4);
    float R = 0.5 * (x * x + y * y + z * z - cr);

    float sigma_esq = maxf(1 - 2 * eps, 0);

    float F = eps * R;

    float dF_x = eps * x;
    float dF_y = eps * y;
    float dF_z = eps * z;

    float M = sqrtf(F * F + sigma_esq * G * G);
    float e_dx = dF_x + (F * dF_x + sigma_esq * G * dG_x) / M;
    float e_dy = dF_y + (F * dF_y + sigma_esq * G * dG_y) / M;
    float e_dz = dF_z + (F * dF_z + sigma_esq * G * dG_z) / M;

    Vector3f pi_transformed;
    pi_transformed[0] = x - e_dx;
    pi_transformed[1] = y - e_dy;
    pi_transformed[2] = z - e_dz;

    ProjResult pr;

    pr.dist = sqrtf(2 * (F + M));
    pr.proj = Q * pi_transformed + pc;

    return pr;
}

inline Matrix3f projection_box_jacobian(float lx, float ly, float lz,
                                        Matrix4f htm, Vector3f point,
                                        float h, float eps)
{
    Vector3f pc = htm.block<3, 1>(0, 3);
    Matrix3f Q = htm.block<3, 3>(0, 0);

    Vector3f q = Q.transpose() * (point - pc);

    float x = q[0];
    float y = q[1];
    float z = q[2];

    float G_x = smf(x - lx / 2, 0, h) + smf(-x - lx / 2, 0, h);
    float G_y = smf(y - ly / 2, 0, h) + smf(-y - ly / 2, 0, h);
    float G_z = smf(z - lz / 2, 0, h) + smf(-z - lz / 2, 0, h);

    float G = G_x + G_y + G_z;

    Vector3f dG;
    dG[0] = smf(x - lx / 2, 1, h) - smf(-x - lx / 2, 1, h);
    dG[1] = smf(y - ly / 2, 1, h) - smf(-y - ly / 2, 1, h);
    dG[2] = smf(z - lz / 2, 1, h) - smf(-z - lz / 2, 1, h);

    Matrix3f HG = Matrix3f::Zero();
    HG(0, 0) = smf(x - lx / 2, 2, h) + smf(-x - lx / 2, 2, h);
    HG(1, 1) = smf(y - ly / 2, 2, h) + smf(-y - ly / 2, 2, h);
    HG(2, 2) = smf(z - lz / 2, 2, h) + smf(-z - lz / 2, 2, h);

    float cr = 1.2 * (lx * lx / 4 + ly * ly / 4 + lz * lz / 4);
    float R = 0.5 * (x * x + y * y + z * z - cr);

    float sigma_esq = maxf(1 - 2 * eps, 0);

    float F = eps * R;

    Vector3f dF = eps * q;
    Matrix3f HF = eps * Matrix3f::Identity();

    float M = sqrtf(F * F + sigma_esq * G * G);

    Vector3f dA = 2 * F * dF + 2 * sigma_esq * G * dG;

    Matrix3f HA =
        2 * (dF * dF.transpose())
        + 2 * F * HF
        + 2 * sigma_esq * (dG * dG.transpose() + G * HG);

    Matrix3f HM =
        (0.5f / M) * HA
        - (0.25f / (M * M * M)) * (dA * dA.transpose());

    Matrix3f HE = HF + HM;

    Matrix3f J_body = Matrix3f::Identity() - HE;

    return Q * J_body * Q.transpose();
}


ProjResult projection_cylinder(float radius, float height, Matrix4f htm, Vector3f point, float h, float eps)
{
    Vector3f pc = htm.block<3, 1>(0, 3);
    Matrix3f Q = htm.block<3, 3>(0, 0);
    Vector3f point_transformed = Q.transpose() * (point - pc);
    float x = point_transformed[0];
    float y = point_transformed[1];
    float z = point_transformed[2];

    float radius_xy = sqrtf(x * x + y * y);
    float delta_radius = radius_xy - radius;

    float G_r = smf(delta_radius, 0, h);
    float G_z = (smf(z - height / 2, 0, h) + smf(-z - height / 2, 0, h));
    float G = G_r + G_z;

    float dG_r = smf(delta_radius, 1, h);
    float dG_z = (smf(z - height / 2, 1, h) - smf(-z - height / 2, 1, h));

    float cr = 1.2 * (2 * radius * radius + height * height / 4);
    float R = 0.5 * (x * x + y * y + z * z - cr);

    float sigma_esq = maxf(1 - 2 * eps, 0);

    float F = eps * R;

    float dF_x = eps * x;
    float dF_y = eps * y;
    float dF_z = eps * z;

    float M = sqrtf(F * F + sigma_esq * G * G);
    float e_dx = dF_x + (F * dF_x + sigma_esq * G * dG_r * (x / radius_xy)) / M;
    float e_dy = dF_y + (F * dF_y + sigma_esq * G * dG_r * (y / radius_xy)) / M;
    float e_dz = dF_z + (F * dF_z + sigma_esq * G * dG_z) / M;

    Vector3f pi_transformed;
    pi_transformed[0] = x - e_dx;
    pi_transformed[1] = y - e_dy;
    pi_transformed[2] = z - e_dz;

    ProjResult pr;

    pr.dist = sqrtf(2 * (F + M));
    pr.proj = Q * pi_transformed + pc;
    return pr;
}

inline Matrix3f projection_cylinder_jacobian(float radius, float height,
                                             Matrix4f htm, Vector3f point,
                                             float h, float eps)
{
    Vector3f pc = htm.block<3, 1>(0, 3);
    Matrix3f Q = htm.block<3, 3>(0, 0);

    Vector3f q = Q.transpose() * (point - pc);

    float x = q[0];
    float y = q[1];
    float z = q[2];

    float r = sqrtf(x * x + y * y);


    float delta_radius = r - radius;

    float G_r = smf(delta_radius, 0, h);
    float G_z = smf(z - height / 2, 0, h) + smf(-z - height / 2, 0, h);

    float G = G_r + G_z;

    float dG_r  = smf(delta_radius, 1, h);
    float ddG_r = smf(delta_radius, 2, h);

    float dG_z  = smf(z - height / 2, 1, h) - smf(-z - height / 2, 1, h);
    float ddG_z = smf(z - height / 2, 2, h) + smf(-z - height / 2, 2, h);

    Vector3f dG;
    dG[0] = dG_r * x / r;
    dG[1] = dG_r * y / r;
    dG[2] = dG_z;

    Matrix3f HG = Matrix3f::Zero();

    Vector2f u;
    u << x / r, y / r;

    Matrix2f I2 = Matrix2f::Identity();

    Matrix2f Hr =
        ddG_r * (u * u.transpose())
        + (dG_r / r) * (I2 - u * u.transpose());

    HG(0, 0) = Hr(0, 0);
    HG(0, 1) = Hr(0, 1);
    HG(1, 0) = Hr(1, 0);
    HG(1, 1) = Hr(1, 1);
    HG(2, 2) = ddG_z;

    float cr = 1.2 * (2 * radius * radius + height * height / 4);
    float R = 0.5 * (x * x + y * y + z * z - cr);

    float sigma_esq = maxf(1 - 2 * eps, 0);

    float F = eps * R;

    Vector3f dF = eps * q;
    Matrix3f HF = eps * Matrix3f::Identity();

    float M = sqrtf(F * F + sigma_esq * G * G);


    Vector3f dA = 2 * F * dF + 2 * sigma_esq * G * dG;

    Matrix3f HA =
        2 * (dF * dF.transpose())
        + 2 * F * HF
        + 2 * sigma_esq * (dG * dG.transpose() + G * HG);

    Matrix3f HM =
        (0.5f / M) * HA
        - (0.25f / (M * M * M)) * (dA * dA.transpose());

    Matrix3f HE = HF + HM;

    Matrix3f J_body = Matrix3f::Identity() - HE;

    return Q * J_body * Q.transpose();
}

ProjResult projection_pointcloud(KDTree tree, PointCloud pc, Vector3f point, float h, float eps)
{
    if (h < 1e-5 && eps < 1e-5)
    {
        float query_pt[3] = {point[0], point[1], point[2]};
        const size_t num_results = 1;
        size_t ret_index;
        float out_dist_sqr;
        nanoflann::KNNResultSet<float> resultSet(num_results);
        resultSet.init(&ret_index, &out_dist_sqr);
        tree->findNeighbors(resultSet, &query_pt[0]);

        ProjResult pr;
        pr.dist = sqrtf(out_dist_sqr);
        pr.proj = Vector3f(pc->kdtree_get_pt(ret_index, 0), pc->kdtree_get_pt(ret_index, 1), pc->kdtree_get_pt(ret_index, 2));

        return pr;
    }
    else
    {

        float min_dist = projection_pointcloud(tree, pc, point, 0, 0).dist;
        float tol = 1e-3;
        float threshold = min_dist / pow(tol, h);

        float query_pt[3] = {point[0], point[1], point[2]};
        float sq_radius = threshold * threshold;
        std::vector<nanoflann::ResultItem<size_t, float>> indices_dists;
        nanoflann::RadiusResultSet<float, size_t> resultSet(sq_radius, indices_dists);
        tree->findNeighbors(resultSet, query_pt);

        float x, y, z, dist;
        vector<Vector3f> all_points;
        vector<float> all_dist;
        min_dist = VERYBIGNUMBER;
        for (int i = 0; i < indices_dists.size(); i++)
        {
            x = pc->kdtree_get_pt(indices_dists[i].first, 0);
            y = pc->kdtree_get_pt(indices_dists[i].first, 1);
            z = pc->kdtree_get_pt(indices_dists[i].first, 2);
            Vector3f new_point = Vector3f(x, y, z);
            dist = (new_point - point).norm();
            min_dist = minf(min_dist, dist);
            all_dist.push_back(dist);
            all_points.push_back(new_point);
        }

        ProjResult pr;
        float sum_weight = 0;
        float weight0, weight1, weight2;
        pr.proj = Vector3f(0, 0, 0);
        float H = 1.0 / h;

        for (int i = 0; i < all_points.size(); i++)
        {
            weight0 = (VERYSMALLNUMBER + min_dist) / (VERYSMALLNUMBER + all_dist[i]);
            weight1 = pow(weight0, H);
            weight2 = weight0 * weight1;
            sum_weight += weight1;
            pr.proj += weight2 * all_points[i];
        }

        float normalization = pow(sum_weight, 1 + h);
        pr.proj = pr.proj / normalization;
        pr.dist = min_dist / pow(sum_weight, h);

        return pr;
    }
}

ProjResult projection_convexpolytope(MatrixXf A, VectorXf b, Matrix4f htm, Vector3f point, float lx, float ly, float lz, Vector3f center, float h, float eps)
{
    ProjResult pr;
    Vector3f pc = htm.block<3, 1>(0, 3);
    Matrix3f Q = htm.block<3, 3>(0, 0);
    Vector3f point_transformed = Q.transpose() * (point - pc);
    Vector3f pi_transformed;

    if (h < 1e-5 && eps < 1e-5)
    {
        pi_transformed = solveQP(Matrix3f::Identity(), -point_transformed, -A, -b);
        pr.dist = (pi_transformed - point_transformed).norm();
    }
    else
    {
        float G = 0;
        Vector3f grad_G = Vector3f(0, 0, 0);
        float inner;
        int N = A.rows();

        for (int i = 0; i < A.rows(); i++)
        {
            inner = A.row(i) * point_transformed - b[i];
            G += smf(inner, 0, h);
            grad_G += smf(inner, 1, h) * A.row(i).transpose();
        }

        // This, ideally, should be computed automatically
        int Nc = N / 2 + 1;

        G = G / Nc;
        grad_G = grad_G / Nc;

        float cr = 1.2 * (lx * lx / 4 + ly * ly / 4 + lz * lz / 4);
        float R = 0.5 * ((point_transformed - center).squaredNorm() - cr);

        float sigma_esq = maxf(1 - 2 * eps, 0);

        float F = eps * R;

        Vector3f grad_F = eps * (point_transformed - center);

        float M = sqrtf(F * F + sigma_esq * G * G);
        Vector3f grad_e = grad_F + (F * grad_F + sigma_esq * G * grad_G) / M;

        pi_transformed = point_transformed - grad_e;
        pr.dist = sqrtf(2 * (F + M));
    }

    pr.proj = Q * pi_transformed + pc;

    return pr;
}

GeometricPrimitives GeometricPrimitives::to_pointcloud(float disc) const
{
    if (type == 0)
        return generate_point_cloud_sphere(lx, htm, disc);
    if (type == 1)
        return generate_point_cloud_box(lx, ly, lz, htm, disc);
    if (type == 2)
        return generate_point_cloud_cylinder(lx, lz, htm, disc);
    if (type == 3)
        return this->copy();
    if (type == 4)
        return generate_point_cloud_convexpolygon(points_gp, A, b, htm, disc);
}

ProjResult GeometricPrimitives::projection(Vector3f point, float h, float eps) const
{
    if (type == 0)
        return projection_sphere(lx, htm, point, h == 0 ? 1e-8 : h, eps == 0 ? 1e-8 : eps);
    if (type == 1)
        return projection_box(lx, ly, lz, htm, point, h == 0 ? 1e-8 : h, eps == 0 ? 1e-8 : eps);
    if (type == 2)
        return projection_cylinder(lx, lz, htm, point, h == 0 ? 1e-8 : h, eps == 0 ? 1e-8 : eps);
    if (type == 3)
        return projection_pointcloud(kdtree, pointcloud, point, h == 0 ? 1e-8 : h, eps == 0 ? 1e-8 : eps);
    if (type == 4)
        return projection_convexpolytope(A, b, htm, point, lx, ly, lz, center, h == 0 ? 1e-8 : h, eps == 0 ? 1e-8 : eps);
}

Matrix3f GeometricPrimitives::projection_jacobian(Vector3f point, float h, float eps) const
{
    if (type == 0)
        return projection_sphere_jacobian(lx, htm, point, h == 0 ? 1e-8 : h, eps == 0 ? 1e-8 : eps);
    if (type == 1)
        return projection_box_jacobian(lx, ly, lz, htm, point, h == 0 ? 1e-8 : h, eps == 0 ? 1e-8 : eps);
    if (type == 2)
        return projection_cylinder_jacobian(lx, lz, htm, point, h == 0 ? 1e-8 : h, eps == 0 ? 1e-8 : eps);
    if (type == 3)
        return Matrix3f::Identity();
    if (type == 4)
        return Matrix3f::Identity();
}

Vector3f support_sphere(Vector3f direction, float radius, Matrix4f htm)
{
    Vector3f pc = htm.block<3, 1>(0, 3);

    return pc + direction.normalized() * radius;
}

Vector3f support_box(Vector3f direction, float lx, float ly, float lz, Matrix4f htm)
{
    Vector3f pc = htm.block<3, 1>(0, 3);
    Matrix3f Q = htm.block<3, 3>(0, 0);
    Vector3f halfExtents(lx / 2, ly / 2, lz / 2);

    Eigen::Vector3f localDir = Q.transpose() * direction;
    Eigen::Vector3f localSupport = halfExtents.cwiseProduct(localDir.array().sign().matrix());
    return pc + Q * localSupport;
}

Vector3f support_cylinder(Vector3f direction, float radius, float height, Matrix4f htm)
{
    Vector3f pc = htm.block<3, 1>(0, 3);
    Matrix3f Q = htm.block<3, 3>(0, 0);

    Eigen::Vector3f localDir = Q.transpose() * direction;

    float dx = localDir[0];
    float dy = localDir[1];
    float dz = localDir[2];
    float r = sqrtf(dx * dx + dy * dy);

    Eigen::Vector3f localSupport;

    if (r > 1e-6)
    {
        localSupport[0] = radius * dx / r;
        localSupport[1] = radius * dy / r;
    }
    else
    {
        localSupport[0] = 0;
        localSupport[1] = 0;
    }

    localSupport[2] = dz > 0 ? height / 2 : -height / 2;

    return pc + Q * localSupport;
}

Vector3f support_convexpolygon(Vector3f direction, vector<Vector3f> points, Matrix4f htm)
{
    float max_value = -VERYBIGNUMBER;
    Vector3f point_selected;
    float aux;

    Vector3f pc = htm.block<3, 1>(0, 3);
    Matrix3f Q = htm.block<3, 3>(0, 0);
    Vector3f pointmod;

    for (int i = 0; i < points.size(); i++)
    {
        pointmod = Q * points[i] + pc;
        aux = direction.dot(pointmod);
        if (aux > max_value)
        {
            max_value = aux;
            point_selected = pointmod;
        }
    }

    return point_selected;
}

Vector3f support_pointcloud(Vector3f direction, vector<Vector3f> points)
{
    float max_value = -VERYBIGNUMBER;
    Vector3f point_selected;
    float aux;

    for (int i = 0; i < points.size(); i++)
    {
        aux = direction.dot(points[i]);
        if (aux > max_value)
        {
            max_value = aux;
            point_selected = points[i];
        }
    }

    return point_selected;
}

Vector3f GeometricPrimitives::support(Vector3f direction) const
{
    if (type == 0)
        return support_sphere(direction, lx, htm);
    if (type == 1)
        return support_box(direction, lx, ly, lz, htm);
    if (type == 2)
        return support_cylinder(direction, lx, lz, htm);
    if (type == 3)
        return support_pointcloud(direction, points_gp);
    if (type == 4)
        return support_convexpolygon(direction, points_gp, htm);
}

float max4(float a1, float a2, float a3, float a4)
{
    return max(a1, max(a2, max(a3, a4)));
}

// AABB

AABB::AABB() {}

AABB AABB::get_aabb_pointcloud(const vector<Vector3f> &points, int start, int end)
{
    Vector3f minPoint = points[start];
    Vector3f maxPoint = points[start];

    for (int i = start; i < end; i++)
    {
        minPoint = minPoint.cwiseMin(points[i]);
        maxPoint = maxPoint.cwiseMax(points[i]);
    }

    AABB box;
    box.p = (minPoint + maxPoint) / 2.0f;
    box.lx = (maxPoint.x() - minPoint.x());
    box.ly = (maxPoint.y() - minPoint.y());
    box.lz = (maxPoint.z() - minPoint.z());

    return box;
}

AABB GeometricPrimitives::get_aabb() const
{

    Vector3f x = htm.block(0, 0, 3, 1);
    Vector3f y = htm.block(0, 1, 3, 1);
    Vector3f z = htm.block(0, 2, 3, 1);

    AABB aabb;

    if (type == 0)
    {
        aabb.lx = 2 * this->lx;
        aabb.ly = 2 * this->lx;
        aabb.lz = 2 * this->lx;
        aabb.p = htm.block(0, 3, 3, 1);

        return aabb;
    }

    if (type == 1)
    {

        Vector3f p1 = this->lx * x + this->ly * y + this->lz * z;
        Vector3f p2 = -this->lx * x + this->ly * y + this->lz * z;
        Vector3f p3 = this->lx * x - this->ly * y + this->lz * z;
        Vector3f p4 = this->lx * x + this->ly * y - this->lz * z;

        float lx = max4(abs(p1[0]), abs(p2[0]), abs(p3[0]), abs(p4[0]));
        float ly = max4(abs(p1[1]), abs(p2[1]), abs(p3[1]), abs(p4[1]));
        float lz = max4(abs(p1[2]), abs(p2[2]), abs(p3[2]), abs(p4[2]));

        aabb.lx = lx;
        aabb.ly = ly;
        aabb.lz = lz;
        aabb.p = htm.block(0, 3, 3, 1);

        return aabb;
    }

    if (type == 2)
    {
        Vector3f p1 = 2 * this->lx * x + 2 * this->lx * y + this->lz * z;
        Vector3f p2 = -2 * this->lx * x + 2 * this->lx * y + this->lz * z;
        Vector3f p3 = 2 * this->lx * x - 2 * this->lx * y + this->lz * z;
        Vector3f p4 = 2 * this->lx * x + 2 * this->lx * y - this->lz * z;

        float lx = max4(abs(p1[0]), abs(p2[0]), abs(p3[0]), abs(p4[0]));
        float ly = max4(abs(p1[1]), abs(p2[1]), abs(p3[1]), abs(p4[1]));
        float lz = max4(abs(p1[2]), abs(p2[2]), abs(p3[2]), abs(p4[2]));

        aabb.lx = lx;
        aabb.ly = ly;
        aabb.lz = lz;
        aabb.p = htm.block(0, 3, 3, 1);

        return aabb;
    }

    if (type == 3)
    {
        aabb.lx = this->lx;
        aabb.ly = this->ly;
        aabb.lz = this->lz;
        aabb.p = this->center;

        return aabb;
    }

    if (type == 4)
    {
        Vector3f p1 = this->lx * x + this->ly * y + this->lz * z;
        Vector3f p2 = -this->lx * x + this->ly * y + this->lz * z;
        Vector3f p3 = this->lx * x - this->ly * y + this->lz * z;
        Vector3f p4 = this->lx * x + this->ly * y - this->lz * z;

        float lx = max4(abs(p1[0]), abs(p2[0]), abs(p3[0]), abs(p4[0]));
        float ly = max4(abs(p1[1]), abs(p2[1]), abs(p3[1]), abs(p4[1]));
        float lz = max4(abs(p1[2]), abs(p2[2]), abs(p3[2]), abs(p4[2]));

        aabb.lx = lx;
        aabb.ly = ly;
        aabb.lz = lz;

        Matrix3f Q = htm.block<3, 3>(0, 0);
        Vector3f p = htm.block<3, 1>(0, 3);

        aabb.p = Q * this->center + p;

        return aabb;
    }
}

float AABB::dist_aabb(AABB aabb1, AABB aabb2)
{
    Vector3f hl1(aabb1.lx / 2, aabb1.ly / 2, aabb1.lz / 2);
    Vector3f hl2(aabb2.lx / 2, aabb2.ly / 2, aabb2.lz / 2);

    Vector3f p1_inf = aabb1.p - hl1;
    Vector3f p1_sup = aabb1.p + hl1;
    Vector3f p2_inf = aabb2.p - hl2;
    Vector3f p2_sup = aabb2.p + hl2;

    return Vector3f(0, 0, 0).cwiseMax(p2_inf - p1_sup).cwiseMax(p1_inf - p2_sup).norm();
}

// BVH

BVH::BVH() {}

int BVH::build_bvh(BVH &bvh, vector<Vector3f> &points, int start, int end, int parentIndex)
{
    AABB nodeAABB = AABB::get_aabb_pointcloud(points, start, end);

    int nodeIndex = bvh.aabb.size();

    bvh.aabb.push_back(nodeAABB);
    bvh.parent.push_back(parentIndex);
    bvh.left_child.push_back(-1);
    bvh.right_child.push_back(-1);

    if (end - start == 1)
    {
        bvh.aabb[nodeIndex].p = points[start];
        return nodeIndex;
    }

    Vector3f size(bvh.aabb[nodeIndex].lx, bvh.aabb[nodeIndex].ly, bvh.aabb[nodeIndex].lz);
    int axis = 0;
    if (size.y() > size.x())
        axis = 1;
    if (size.z() > size[axis])
        axis = 2;

    sort(points.begin() + start, points.begin() + end,
         [axis](const Vector3f &a, const Vector3f &b)
         {
             return a[axis] < b[axis];
         });

    int mid = (start + end) / 2;

    int leftChild = build_bvh(bvh, points, start, mid, nodeIndex);
    int rightChild = build_bvh(bvh, points, mid, end, nodeIndex);

    bvh.left_child[nodeIndex] = leftChild;
    bvh.right_child[nodeIndex] = rightChild;

    //

    //

    return nodeIndex;
}

BVH::BVH(vector<Vector3f> &points)
{
    if (points.empty())
        return;
    build_bvh(*this, points, 0, points.size(), -1);
}

void tvec(Vector3f v, float *vf)
{

    vf[0] = v[0];
    vf[1] = v[1];
    vf[2] = v[2];
}

string pmat(Matrix4f m)
{
    string s = "np.matrix([";

    for (int i = 0; i < 4; i++)
    {
        s += "[";
        for (int j = 0; j < 4; j++)
        {
            s += print_number(m(i, j), 8);
            if (j < 3)
                s += ",";
        }
        if (i < 3)
            s += "],";
        else
            s += "]";
    }
    s += "])";

    return s;
}

QueueElement eval_node(int index, const GeometricPrimitives &prim, const AABB &prim_aabb, const BVH &bvh, PrimDistResult &bestResult)
{
    QueueElement qe;
    qe.nodeIndex = index;

    if (bvh.left_child[index] == -1 && bvh.right_child[index] == -1)
    {
        ProjResult pr = prim.projection(bvh.aabb[index].p, 0, 0);
        qe.dist = pr.dist;
        qe.proj_A = bvh.aabb[index].p;
        qe.proj_B = pr.proj;

        if (pr.dist < bestResult.dist)
        {
            bestResult.dist = pr.dist;
            bestResult.proj_A = bvh.aabb[index].p;
            bestResult.proj_B = pr.proj;
        }
        return qe;
    }
    else
    {
        qe.dist = AABB::dist_aabb(bvh.aabb[index], prim_aabb);
        qe.proj_A = Vector3f(0, 0, 0);
        qe.proj_B = Vector3f(0, 0, 0);
        return qe;
    }
}

PrimDistResult dist_to_bvh(const GeometricPrimitives &prim, const BVH &bvh)
{
    // Min priority queue (min-heap) for best-first traversal
    priority_queue<QueueElement, vector<QueueElement>, greater<QueueElement>> pq;

    // Initialize best distance
    PrimDistResult bestResult;
    bestResult.dist = VERYBIGNUMBER;

    // Start with the root node
    if (bvh.aabb.empty())
        return bestResult;

    AABB prim_aabb = prim.get_aabb();

    pq.push(eval_node(0, prim, prim_aabb, bvh, bestResult)); // Root node

    while (!pq.empty())
    {
        // Get the closest node from the queue
        QueueElement current = pq.top();
        pq.pop();

        int nodeIndex = current.nodeIndex;

        // Prune
        if (current.dist >= bestResult.dist)
            continue;

        // Otherwise
        if (bvh.left_child[nodeIndex] != -1)
            pq.push(eval_node(bvh.left_child[nodeIndex], prim, prim_aabb, bvh, bestResult));
        if (bvh.right_child[nodeIndex] != -1)
            pq.push(eval_node(bvh.right_child[nodeIndex], prim, prim_aabb, bvh, bestResult));
    }

    return bestResult;
}

QueueElement eval_node_range(int index, const GeometricPrimitives &prim, const AABB &prim_aabb, const BVH &bvh, vector<Vector3f> &result, float threshold)
{
    QueueElement qe;
    qe.nodeIndex = index;

    if (bvh.left_child[index] == -1 && bvh.right_child[index] == -1)
    {
        ProjResult pr = prim.projection(bvh.aabb[index].p, 0, 0);
        qe.dist = pr.dist;
        qe.proj_A = bvh.aabb[index].p;
        qe.proj_B = pr.proj;

        if (pr.dist <= threshold)
            result.push_back(bvh.aabb[index].p);

        return qe;
    }
    else
    {
        qe.dist = AABB::dist_aabb(bvh.aabb[index], prim_aabb);
        qe.proj_A = Vector3f(0, 0, 0);
        qe.proj_B = Vector3f(0, 0, 0);
        return qe;
    }
}

vector<Vector3f> dist_to_bvh_range(const GeometricPrimitives &prim, const BVH &bvh, float threshold)
{
    vector<Vector3f> result = {}; // Stores the points in leaf nodes that satisfy the condition

    // Min priority queue (min-heap) for best-first traversal
    priority_queue<QueueElement, vector<QueueElement>, greater<QueueElement>> pq;

    // Initialize best distance
    PrimDistResult bestResult;
    bestResult.dist = VERYBIGNUMBER;

    AABB prim_aabb = prim.get_aabb();

    // Start with the root node
    if (bvh.aabb.empty())
        return result;

    pq.push(eval_node_range(0, prim, prim_aabb, bvh, result, threshold)); // Root node

    while (!pq.empty())
    {
        // Get the closest node from the queue
        QueueElement current = pq.top();
        pq.pop();

        int nodeIndex = current.nodeIndex;

        // Prune
        if (current.dist > threshold)
            continue;

        // cout << "Creating descendants..."<<std::endl;
        // Otherwise
        if (bvh.left_child[nodeIndex] != -1)
            pq.push(eval_node_range(bvh.left_child[nodeIndex], prim, prim_aabb, bvh, result, threshold));
        if (bvh.right_child[nodeIndex] != -1)
            pq.push(eval_node_range(bvh.right_child[nodeIndex], prim, prim_aabb, bvh, result, threshold));
    }

    return result;
}

PrimDistResult dist_to_bvh_smooth(const GeometricPrimitives &prim, int pc_size, const BVH &bvh, float h, float eps)
{
    PrimDistResult pdr;

    float min_dist = dist_to_bvh(prim, bvh).dist;

    float tol = 0.05; // 1e-3
    float threshold = min_dist / pow(tol, h);

    vector<Vector3f> all_points = dist_to_bvh_range(prim, bvh, threshold);
    vector<ProjResult> all_dist;

    float min_dist_smooth = VERYBIGNUMBER;
    float aux_dist;

    // cout << "Usage = "<<((float) all_points.size())/((float) pc_size)<<std::endl;

    for (int i = 0; i < all_points.size(); i++)
    {
        ProjResult proj_aux = prim.projection(all_points[i], h, eps);
        all_dist.push_back(proj_aux);
        min_dist_smooth = minf(min_dist_smooth, proj_aux.dist);
    }

    float sum_weight = 0;
    float weight0, weight1, weight2;
    pdr.proj_A = Vector3f(0, 0, 0);
    pdr.proj_B = Vector3f(0, 0, 0);
    pdr.aux = Vector3f(0, 0, 0);
    pdr.dist = 0;
    float H = 1.0 / h;

    min_dist_smooth = 0.5 * pow(min_dist_smooth, 2);

    for (int i = 0; i < all_points.size(); i++)
    {
        weight0 = (VERYSMALLNUMBER + min_dist_smooth) / (VERYSMALLNUMBER + 0.5 * pow(all_dist[i].dist, 2));
        weight1 = pow(weight0, H);
        weight2 = weight0 * weight1;
        sum_weight += weight1;
        pdr.proj_A += weight2 * all_points[i];
        pdr.proj_B += weight2 * all_dist[i].proj;
        pdr.aux += weight2 * all_dist[i].proj.cross(all_points[i]);
    }

    float normalization = pow(sum_weight, 1 + h);
    pdr.proj_A = pdr.proj_A / normalization;
    pdr.proj_B = pdr.proj_B / normalization;
    pdr.aux = pdr.aux / normalization;

    pdr.dist = sqrtf(2 * min_dist_smooth / pow(sum_weight, h));

    return pdr;
}

PrimDistResult dist_to_gjk(const GeometricPrimitives &objA, const GeometricPrimitives &objB)
{
    // Implemented by https://gist.github.com/vurtun/29727217c269a2fbf4c0ed9a1d11cb40

    /* initial guess */
    bool cont = true;
    struct gjk_support s;
    struct gjk_simplex gsx;

    while (cont)
    {

        Vector3f da = Vector3f::Random();
        Vector3f db = Vector3f::Random();

        s = {0};
        tvec(objA.support(da), s.a);
        tvec(objB.support(db), s.b);

        /* run gjk algorithm */
        int k = 0;
        gsx = {0};
        gsx.error = 0;
        while (gjk(&gsx, &s) && gsx.error == 0)
        {
            da = Vector3f(s.da[0], s.da[1], s.da[2]);
            db = Vector3f(s.db[0], s.db[1], s.db[2]);

            tvec(objA.support(da), s.a);
            tvec(objB.support(db), s.b);

            // Just add some fake point indexes
            s.aid = k;
            s.bid = k + 1;
            k += 2;
        }

        cont = gsx.error == 1;
    }

    /* check distance between closest points */
    struct gjk_result res;
    gjk_analyze(&res, &gsx);

    PrimDistResult pdr;

    pdr.proj_A = Vector3f(res.p0[0], res.p0[1], res.p0[2]);
    pdr.proj_B = Vector3f(res.p1[0], res.p1[1], res.p1[2]);
    pdr.dist = sqrtf(res.distance_squared);

    return pdr;
}

PrimDistResult dist_to_gap(GeometricPrimitives objA, GeometricPrimitives objB, float h, float eps, float tol, int no_iter_max, Vector3f p_A0)
{
    Vector3f p_A = p_A0;
    Vector3f p_B, p_A_old;

    bool cont = true;
    int iter = 0;

    vector<float> hist_error = {};

    while (cont)
    {
        p_B = objB.projection(p_A, h, eps).proj;
        p_A_old = p_A;
        p_A = objA.projection(p_B, h, eps).proj;
        cont = ((p_A - p_A_old).norm() > tol) && (iter <= no_iter_max);

        hist_error.push_back((p_A - p_A_old).norm());
        iter++;
    }

    PrimDistResult pdr;
    float dist_AB = objB.projection(p_A, h, eps).dist;
    float dist_BA = objA.projection(p_B, h, eps).dist;
    float dist_AB_sq = dist_AB * dist_AB;
    float dist_BA_sq = dist_BA * dist_BA;

    pdr.dist = sqrtf(maxf(dist_AB_sq + dist_BA_sq - (p_A - p_B).squaredNorm(), 0.0f));
    pdr.proj_A = p_A;
    pdr.proj_B = p_B;

    pdr.hist_error = hist_error;

    return pdr;
}

PrimDistResult GeometricPrimitives::dist_to(GeometricPrimitives prim, float h, float eps, float tol, int no_iter_max, Vector3f p_A0) const
{
    if (h < 1e-5 && eps < 1e-5)
    {
        if (type != 3 && prim.type != 3)
            // Both are non-point cloud primitives, and no smoothing is required
            // Call GJK
            return dist_to_gjk(*this, prim);
        else
        {

            if (type == 3 && prim.type == 3)
            {
                // Both are point clouds, and no smoothing is required
                // Use brute force with one of the KD-Trees...
                throw std::runtime_error("Not implemented yet (both point clouds with no smoothing)");
            }
            else
            {
                // One of them is a point cloud, and no smoothing is required
                // Call the KDtree-based algorithm

                if (type == 3)
                    return dist_to_bvh(prim, bvh);
                else
                {
                    PrimDistResult pdr = dist_to_bvh(*this, prim.bvh);
                    Vector3f aux = pdr.proj_A;
                    pdr.proj_A = pdr.proj_B;
                    pdr.proj_B = aux;
                    return pdr;
                }
            }
        }
    }
    else
    {
        if (type != 3 && prim.type != 3)
            // Both are non-point cloud primitives, and smoothing is required
            // Call the generalized alternating projection (gap)
            return dist_to_gap(*this, prim, h, eps, tol, no_iter_max, p_A0);
        else
        {

            if (type == 3 && prim.type == 3)
            {
                // Both are point clouds, and smoothing is required
                throw std::runtime_error("Not implemented yet (both pointcloud with smoothing)");
            }
            else
            {
                if (type == 3)
                    return dist_to_bvh_smooth(prim, points_gp.size(), bvh, h, eps);
                else
                {
                    PrimDistResult pdr = dist_to_bvh_smooth(*this, prim.points_gp.size(), prim.bvh, h, eps);
                    Vector3f aux = pdr.proj_A;
                    pdr.proj_A = pdr.proj_B;
                    pdr.proj_B = aux;
                    return pdr;
                }
            }
        }
    }
}

PrimDistResult GeometricPrimitives::dist_to(GeometricPrimitives prim, float h, float eps, float tol, int no_iter_max) const
{
    return dist_to(prim, h, eps, tol, no_iter_max, htm.block<3, 1>(0, 3));
}

GeometricPrimitives GeometricPrimitives::copy() const
{
    if (type == 0)
        return GeometricPrimitives::create_sphere(htm, lx);
    if (type == 1)
        return GeometricPrimitives::create_box(htm, lx, ly, lz);
    if (type == 2)
        return GeometricPrimitives::create_cylinder(htm, lx, lz);
    if (type == 3)
    {
        vector<Vector3f> points = points_gp;
        return GeometricPrimitives::create_pointcloud(points);
    }
    if (type == 4)
        return GeometricPrimitives::create_convexpolytope(htm, A, b);
}

string PrimDistResult::toString() const
{
    std::ostringstream oss;
    oss << "PRIMITIVE DISTANCE RESULT" << std::endl;
    oss << "Distance: " << print_number(dist) << " m" << std::endl;
    oss << "Projection A: " << print_vector(proj_A) << " m" << std::endl;
    oss << "Projection B: " << print_vector(proj_B) << " m" << std::endl;

    return oss.str();
}

string ProjResult::toString() const
{
    std::ostringstream oss;
    oss << "PROJECTION PRIMITIVE" << std::endl;
    oss << "Distance: " << print_number(dist) << " m" << std::endl;
    oss << "Projection: " << print_vector(proj) << " m" << std::endl;

    return oss.str();
}

////////////////////////////////////////////////////////////////
// MANIPULATOR
////////////////////////////////////////////////////////////////

string print_no_link(int ind_link)
{
    int dig_tot = 3;
    int dig = (ind_link == 0) ? 1 : 1 + (int)floor(log10((float)ind_link));

    string str = std::to_string(ind_link);
    for (int i = 0; i < dig_tot - dig; i++)
        str += " ";

    return str;
}

string Manipulator::toString() const
{
    std::ostringstream oss;
    oss << "MANIPULATOR" << std::endl;
    oss << "Degrees of freedom : " << no_links << std::endl;
    oss << "No primitives: " << geo_prim.size() << std::endl;
    oss << "q_min: " << print_vector(q_min) << " rad" << std::endl;
    oss << "q_max: " << print_vector(q_max) << " rad" << std::endl;

    oss << "DH Table (Standard): " << std::endl;
    oss << "          theta(rad)   d(m)    alpha(rad)    a(m)" << std::endl;
    for (int ind_link = 0; ind_link < no_links; ind_link++)
    {
        oss << "Link " << print_no_link(ind_link) << ": ";
        oss << (joint_type[ind_link] == 0 ? "   V   " : print_number(theta[ind_link])) << "   ";
        oss << (joint_type[ind_link] == 1 ? "   V   " : print_number(dh_d[ind_link])) << "   ";
        oss << print_number(alpha[ind_link]) << "   ";
        oss << print_number(dh_a[ind_link]) << "  " << std::endl;
    }

    oss << "Transformation from world to first DH:" << std::endl;
    oss << print_matrix(htm_world_to_dh0) << std::endl;
    oss << "Transformation from last DH to EE:" << std::endl;
    oss << print_matrix(htm_dhn_to_ee) << std::endl;

    return oss.str();
}

Manipulator::Manipulator(int _no_links) : no_links(_no_links)
{
    for (int no_link = 0; no_link < no_links; no_link++)
    {
        theta.push_back(0);
        dh_d.push_back(0);
        alpha.push_back(0);
        dh_a.push_back(0);
        joint_type.push_back(0);
        dh_cos_theta.push_back(1.0);
        dh_sin_theta.push_back(0.0);
        dh_cos_alpha.push_back(1.0);
        dh_sin_alpha.push_back(0.0);
    }

    htm_world_to_dh0 = Matrix4f::Identity();
    htm_dhn_to_ee = Matrix4f::Identity();

    no_tube_points = 0;
    for (int ind_link = 0; ind_link < no_links; ind_link++)
        coord_tube.push_back({});

    no_prim = 0;
    for (int ind_link = 0; ind_link < no_links; ind_link++)
        geo_prim.push_back({});

    tube_radius = 0;

    q_min = -M_PI * VectorXf::Ones(no_links);
    q_max = M_PI * VectorXf::Ones(no_links);
}

void Manipulator::set_joint_param(int ind_link, float _theta, float _d, float _alpha, float _a, int _joint_type, float _q_min, float _q_max)
{
    if (ind_link < 0 || ind_link >= no_links)
        throw std::runtime_error("The link index should be between 0 and " + std::to_string(no_links - 1) + "!");
    if (_joint_type != 0 && _joint_type != 1)
        throw std::runtime_error("The joint type should be 0 (rotative) or 1 (prismatic)!");
    if (_q_min >= _q_max)
        throw std::runtime_error("q_max should be strictly greater than q_min!");

    theta[ind_link] = _theta;
    dh_d[ind_link] = _d;
    alpha[ind_link] = _alpha;
    dh_a[ind_link] = _a;
    joint_type[ind_link] = _joint_type;
    dh_cos_theta[ind_link] = cosf(_theta);
    dh_sin_theta[ind_link] = sinf(_theta);
    dh_cos_alpha[ind_link] = cosf(_alpha);
    dh_sin_alpha[ind_link] = sin(_alpha);
    q_min[ind_link] = _q_min;
    q_max[ind_link] = _q_max;
}

void Manipulator::add_tube_coord(int ind_link, Vector3f coord)
{
    if (ind_link < 0 || ind_link >= no_links)
        throw std::runtime_error("The link index should be between 0 and " + std::to_string(no_links - 1) + "!");

    int last_ind_link = -1;

    for (int ind_link = 0; ind_link < no_links; ind_link++)
        if (coord_tube[ind_link].size() > 0)
            last_ind_link = ind_link;

    if (last_ind_link >= 0)
    {
        if (ind_link < last_ind_link)
            throw std::runtime_error("A point in the tube has link index " + std::to_string(ind_link) + ", but the previous point has link index " + std::to_string(last_ind_link) + ". Subsequent points should be either in the same link or subsequent links in the kinematic chain!");

        if (ind_link - last_ind_link > 1)
            cout << "WARNING!!! A point in the tube has link index " << ind_link << ", but the previous point has link index " << last_ind_link << ". Gradient computation is not supported when this index difference is greater than 1!" << std::endl;
        else
        {
            if (ind_link - last_ind_link == 1)
            {
                Vector3f last_coord = coord_tube[last_ind_link][coord_tube[last_ind_link].size() - 1];
                if ((abs(last_coord[0]) > 1.0 / VERYBIGNUMBER || abs(last_coord[1]) > 1.0 / VERYBIGNUMBER))
                    cout << "WARNING!!! The previous tube has point has local coordinates that have non-zero x and y coordinates, while the current point is in a different link. Gradient computation is not supported in this case!" << std::endl;
            }
        }
    }

    if (no_tube_points == 0)
        no_tube_points = 2;
    else
        no_tube_points++;

    coord_tube[ind_link].push_back(coord);
}

void Manipulator::add_geo_prim(int ind_link, GeometricPrimitives prim)
{
    if (ind_link < 0 || ind_link >= no_links)
        throw std::runtime_error("The link index should be between 0 and " + std::to_string(no_links - 1) + "!");

    no_prim++;

    geo_prim[ind_link].push_back(prim);
}

void Manipulator::set_htm_extra(Matrix4f _htm_world_to_dh0, Matrix4f _htm_dhn_to_ee)
{
    htm_world_to_dh0 = _htm_world_to_dh0;
    htm_dhn_to_ee = _htm_dhn_to_ee;
}

Manipulator::Manipulator() {};

vector<FKResult> Manipulator::fk(const vector<VectorXf> &q, const vector<Matrix4f> &htm_world_base, bool compute_jac) const
{
    int no_q = q.size();
    vector<FKResult> fkres_all(no_q);

    for (int ind_q = 0; ind_q < no_q; ind_q++)
    {
        // Manipulator manip = *this;
        Matrix4f htm_0 = htm_world_base[ind_q] * htm_world_to_dh0;
        Matrix4f htm = htm_0;
        FKResult fkres(no_links);

        if (q[ind_q].rows() != no_links)
            throw std::runtime_error("The configuration vector q should have " + std::to_string(no_links) + " rows!");

        // Compute the forward kinematic
        for (int ind_links = 0; ind_links < no_links; ind_links++)
        {

            Matrix4f trns = Matrix4f::Identity();
            float c_theta = (joint_type[ind_links] == 0) ? cosf(q[ind_q][ind_links]) : dh_cos_theta[ind_links];
            float s_theta = (joint_type[ind_links] == 0) ? sinf(q[ind_q][ind_links]) : dh_sin_theta[ind_links];
            float d = (joint_type[ind_links] == 1) ? q[ind_q][ind_links] : dh_d[ind_links];
            float c_alpha = dh_cos_alpha[ind_links];
            float s_alpha = dh_sin_alpha[ind_links];
            float a = dh_a[ind_links];

            trns(0, 0) = c_theta;
            trns(0, 1) = -c_alpha * s_theta;
            trns(0, 2) = s_alpha * s_theta;
            trns(0, 3) = a * c_theta;
            trns(1, 0) = s_theta;
            trns(1, 1) = c_alpha * c_theta;
            trns(1, 2) = -c_theta * s_alpha;
            trns(1, 3) = a * s_theta;
            trns(2, 0) = 0;
            trns(2, 1) = s_alpha;
            trns(2, 2) = c_alpha;
            trns(2, 3) = d;

            htm = htm * trns;
            fkres.htm_dh[ind_links] = htm;
        }

        fkres.htm_ee = fkres.htm_dh[no_links - 1] * htm_dhn_to_ee;

        if (compute_jac)
        {
            // Compute the differential kinematics
            for (int ind_links = 0; ind_links < no_links; ind_links++)
            {
                fkres.jac_v_dh[ind_links] = MatrixXf::Zero(3, no_links);
                fkres.jac_w_dh[ind_links] = MatrixXf::Zero(3, no_links);

                Vector3f p_i = fkres.get_p_dh(ind_links);

                for (int ind_cols = 0; ind_cols <= ind_links; ind_cols++)
                {
                    Vector3f p_j_ant;
                    Vector3f z_j_ant;
                    Vector3f v;
                    Vector3f w;

                    p_j_ant = (ind_cols > 0) ? fkres.get_p_dh(ind_cols - 1) : htm_0.block<3, 1>(0, 3);
                    z_j_ant = (ind_cols > 0) ? fkres.get_z_dh(ind_cols - 1) : htm_0.block<3, 1>(0, 2);

                    v = (joint_type[ind_cols] == 0) ? z_j_ant.cross(p_i - p_j_ant) : z_j_ant;
                    w = (joint_type[ind_cols] == 0) ? z_j_ant : Vector3f::Zero();

                    for (int ind_rows = 0; ind_rows < 3; ind_rows++)
                    {
                        fkres.jac_v_dh[ind_links](ind_rows, ind_cols) = v[ind_rows];
                        fkres.jac_w_dh[ind_links](ind_rows, ind_cols) = w[ind_rows];
                    }
                }
            }

            fkres.jac_v_ee = fkres.jac_v_dh[no_links - 1] + s_mat(fkres.get_p_dh(no_links - 1) - fkres.get_p_ee()) * fkres.jac_w_dh[no_links - 1];
            fkres.jac_w_ee = fkres.jac_w_dh[no_links - 1];
        }

        fkres_all[ind_q] = fkres;
    }

    return fkres_all;
}

FKResult Manipulator::fk(VectorXf q, Matrix4f htm_world_base, bool compute_jac) const
{
    vector<VectorXf> vec_q = {q};
    vector<Matrix4f> vec_htm_world_base = {htm_world_base};
    vector<FKResult> fk_res = fk(vec_q, vec_htm_world_base, compute_jac);
    return fk_res[0];
}

vector<FKPrimResult> Manipulator::fk_prim(const vector<VectorXf> &q, const vector<FKResult> &fk_res_all) const
{

    int no_q = q.size();
    vector<FKPrimResult> fkres_prim_all(no_q);

    for (int ind_q = 0; ind_q < no_q; ind_q++)
    {
        FKPrimResult fkres_prim;
        for (int ind_links = 0; ind_links < no_links; ind_links++)
            for (int ind_prim = 0; ind_prim < geo_prim[ind_links].size(); ind_prim++)
            {
                Matrix4f htm_prim = fk_res_all[ind_q].htm_dh[ind_links] * geo_prim[ind_links][ind_prim].htm;
                Vector3f p_prim = htm_prim.block<3, 1>(0, 3);
                Vector3f pc = p_prim - fk_res_all[ind_q].get_p_dh(ind_links);

                fkres_prim.htm_prim.push_back(htm_prim);
                fkres_prim.jac_v_prim.push_back(fk_res_all[ind_q].jac_v_dh[ind_links] - s_mat(pc) * fk_res_all[ind_q].jac_w_dh[ind_links]);
                fkres_prim.jac_w_prim.push_back(fk_res_all[ind_q].jac_w_dh[ind_links]);
            }

        fkres_prim_all[ind_q] = fkres_prim;
    }

    return fkres_prim_all;
}

FKPrimResult Manipulator::fk_prim(VectorXf q, Matrix4f htm_world_base) const
{
    // return fk_prim({q}, {fk(q, htm_world_base)})[0];
    //
    return fk_prim(vector<VectorXf>{q}, vector<FKResult>{fk(q, htm_world_base)})[0];
}

TaskResult Manipulator::fk_task(VectorXf q, Matrix4f htm_world_base, Matrix4f tg_htm) const
{
    int n = no_links;
    FKResult fkres = fk(q, htm_world_base);

    Vector3f p_ee = fkres.get_p_ee();
    Vector3f x_ee = fkres.get_x_ee();
    Vector3f y_ee = fkres.get_y_ee();
    Vector3f z_ee = fkres.get_z_ee();

    Vector3f p_des = tg_htm.block<3, 1>(0, 3);
    Vector3f x_des = tg_htm.block<3, 1>(0, 0);
    Vector3f y_des = tg_htm.block<3, 1>(0, 1);
    Vector3f z_des = tg_htm.block<3, 1>(0, 2);

    Vector3f r_p = p_ee - p_des;
    Vector3f r_o;

    const float EPS = 1e-8;

    float e_x = maxf(minf(1.0 - x_des.transpose() * x_ee, 2.0 - EPS), EPS);
    float e_y = maxf(minf(1.0 - y_des.transpose() * y_ee, 2.0 - EPS), EPS);
    float e_z = maxf(minf(1.0 - z_des.transpose() * z_ee, 2.0 - EPS), EPS);

    r_o << e_x, e_y, e_z;

    VectorXf r(6, 1);
    r << r_p, r_o;

    MatrixXf jac_v = fkres.jac_v_ee;
    MatrixXf jac_w = fkres.jac_w_ee;
    MatrixXf jacr_o_x = x_des.transpose() * s_mat(x_ee) * jac_w;
    MatrixXf jacr_o_y = y_des.transpose() * s_mat(y_ee) * jac_w;
    MatrixXf jacr_o_z = z_des.transpose() * s_mat(z_ee) * jac_w;

    MatrixXf jac_task(6, n);
    jac_task << jac_v, jacr_o_x, jacr_o_y, jacr_o_z;

    float max_error_pos = maxf(abs(r[0]), maxf(abs(r[1]), abs(r[2])));
    float max_error_ori = (180 / M_PI) * maxf(acos(1.0 - r[3]), maxf(acos(1.0 - r[4]), acos(1.0 - r[5])));

    TaskResult tr;

    tr.task = r;
    tr.jac_task = jac_task;
    tr.max_error_pos = max_error_pos;
    tr.max_error_ori = max_error_ori;

    return tr;
}

IKResult Manipulator::ik(Matrix4f tg_htm, Matrix4f htm, VectorXf q0, float p_tol, float a_tol, int no_iter_max, bool ignore_orientation) const
{
    float eta = 0.03;
    float eps = 0.001;
    bool cont = true;
    VectorXf q = q0;
    int no_iter = 0;
    float error_p;
    float error_a;

    const int n = q0.rows();
    TaskResult tr;

    while (cont)
    {
        tr = this->fk_task(q, htm, tg_htm);

        if (!ignore_orientation)
        {
            VectorXf task = tr.task;
            q += dp_inv_solve(tr.jac_task, -eta * sqrt_sign(task), eps);
        }
        else
        {
            q += dp_inv_solve(tr.jac_task.block(0, 0, 3, n), -eta * sqrt_sign(tr.task.block(0, 0, 3, 1)), eps);
            tr.max_error_ori = 0;
        }

        no_iter++;
        cont = (no_iter < no_iter_max) && ((tr.max_error_pos > p_tol) || (tr.max_error_ori > a_tol));
    }

    IKResult ikr;

    ikr.error_pos = tr.max_error_pos;
    ikr.error_ori = tr.max_error_ori;
    ikr.qf = q;
    ikr.success = (tr.max_error_pos <= p_tol) && (tr.max_error_ori <= a_tol);

    return ikr;
}

// Check distances

CheckFreeConfigResult Manipulator::check_free_configuration(VectorXf q, Matrix4f htm, vector<GeometricPrimitives> obstacles, bool check_joint,
                                                            bool check_auto, float tol, float dist_tol, int no_iter_max) const
{
    CheckFreeConfigResult cfcr;

    // Check joint limit
    if (check_joint)
    {
        for (int ind_links = 0; ind_links < no_links; ind_links++)
        {
            if (q[ind_links] < q_min[ind_links])
            {
                cfcr.isfree = false;
                cfcr.message = "Joint number " + std::to_string(ind_links) + " is below minimum limit.";
                cfcr.info.push_back(0);
                cfcr.info.push_back(ind_links);
                return cfcr;
            }
            if (q[ind_links] > q_max[ind_links])
            {
                cfcr.isfree = false;
                cfcr.message = "Joint number " + std::to_string(ind_links) + " is above maximum limit.";
                cfcr.info.push_back(1);
                cfcr.info.push_back(ind_links);
                return cfcr;
            }
        }
    }

    FKResult fkres = fk(q, htm, false);

    vector<GeometricPrimitives> objects_links;
    vector<AABB> objects_links_aabb;
    vector<int> list_ind_links;
    vector<int> list_ind_obj_links;

    for (int ind_links = 0; ind_links < no_links; ind_links++)
        for (int ind_obj_link = 0; ind_obj_link < geo_prim[ind_links].size(); ind_obj_link++)
        {
            GeometricPrimitives obj_copy = geo_prim[ind_links][ind_obj_link].copy();
            obj_copy.htm = fkres.htm_dh[ind_links] * geo_prim[ind_links][ind_obj_link].htm;

            objects_links.push_back(obj_copy);
            list_ind_links.push_back(ind_links);
            list_ind_obj_links.push_back(ind_obj_link);
            objects_links_aabb.push_back(obj_copy.get_aabb());
        }

    // Check collision with obstacles
    for (int ind_tot = 0; ind_tot < objects_links.size(); ind_tot++)
        for (int ind_obst = 0; ind_obst < obstacles.size(); ind_obst++)
        {
            AABB aabb_obs = obstacles[ind_obst].get_aabb();
            int ind_link = list_ind_links[ind_tot];
            int ind_obj_links = list_ind_obj_links[ind_tot];

            if (AABB::dist_aabb(aabb_obs, objects_links_aabb[ind_tot]) == 0)
                if (objects_links[ind_tot].dist_to(obstacles[ind_obst], 1e-6, 1e-6, tol, no_iter_max).dist < dist_tol)
                {

                    cfcr.isfree = false;
                    cfcr.message = "Collision between link " + std::to_string(ind_link) + " (col object " + std::to_string(ind_obj_links) + ") and obstacle " + std::to_string(ind_obst) + ".";
                    cfcr.info.push_back(2);
                    cfcr.info.push_back(ind_link);
                    cfcr.info.push_back(ind_obj_links);
                    cfcr.info.push_back(ind_obst);
                    return cfcr;
                }
        }

    // Check auto collision
    if (check_auto)
    {
        for (int ind_tot_A = 0; ind_tot_A < objects_links.size(); ind_tot_A++)
            for (int ind_tot_B = 0; ind_tot_B < objects_links.size(); ind_tot_B++)
            {
                int ind_link_A = list_ind_links[ind_tot_A];
                int ind_obj_links_A = list_ind_obj_links[ind_tot_A];
                int ind_link_B = list_ind_links[ind_tot_B];
                int ind_obj_links_B = list_ind_obj_links[ind_tot_B];

                if (ind_link_B > ind_link_A + 1)
                {

                    if (AABB::dist_aabb(objects_links_aabb[ind_tot_A], objects_links_aabb[ind_tot_B]) == 0)
                    {

                        if (objects_links[ind_tot_A].dist_to(objects_links[ind_tot_B], 1e-6, 1e-6, tol, no_iter_max).dist < dist_tol)
                        {
                            cfcr.isfree = false;
                            cfcr.message = "Collision between link " + std::to_string(ind_link_A) + " (col object " + std::to_string(ind_obj_links_A) + ") and link " + std::to_string(ind_link_B) + " (col object " + std::to_string(ind_obj_links_B) + ").";
                            cfcr.info.push_back(3);
                            cfcr.info.push_back(ind_link_A);
                            cfcr.info.push_back(ind_obj_links_A);
                            cfcr.info.push_back(ind_link_B);
                            cfcr.info.push_back(ind_obj_links_B);
                            return cfcr;
                        }
                    }
                }
            }
    }

    // Everything is ok!
    cfcr.isfree = true;
    cfcr.message = "Ok!";
    cfcr.info = {};

    return cfcr;
}

DistStructLinkObj::DistStructLinkObj() {};

DistStructLinkObj DistStructRobotObj::get_item(int ind_link, int ind_obj_link)
{
    for (int ind_info = 0; ind_info < list_info.size(); ind_info++)
    {
        if ((list_info[ind_info].link_number == ind_link) && (list_info[ind_info].link_col_obj_number == ind_obj_link))
            return list_info[ind_info];
    }

    DistStructLinkObj dslo;
    dslo.is_null = true;

    return dslo;
}

DistStructRobotObj::DistStructRobotObj() {};

DistStructRobotObj Manipulator::compute_dist(GeometricPrimitives obj, VectorXf q, Matrix4f htm, DistStructRobotObj old_dist_struct,
                                             float tol, int no_iter_max, float max_dist, float h, float eps) const
{
    FKResult fkres = fk(q, htm, true);

    AABB obj_aabb = obj.get_aabb();

    DistStructRobotObj dsro;
    dsro.is_null = false;
    dsro.list_info = {};

    MatrixXf jac_tot = MatrixXf::Zero(0, q.rows());
    VectorXf dist_tot = VectorXf::Zero(0);
    int old_rows;

    for (int ind_links = 0; ind_links < no_links; ind_links++)
    {
        MatrixXf Jv = fkres.jac_v_dh[ind_links];
        MatrixXf Jw = fkres.jac_w_dh[ind_links];
        Vector3f pc = fkres.get_p_dh(ind_links);
        MatrixXf Jvv = Jv + s_mat(pc) * Jw;

        for (int ind_obj_link = 0; ind_obj_link < geo_prim[ind_links].size(); ind_obj_link++)
        {

            GeometricPrimitives obj_copy = geo_prim[ind_links][ind_obj_link].copy();
            obj_copy.htm = fkres.htm_dh[ind_links] * geo_prim[ind_links][ind_obj_link].htm;

            if (AABB::dist_aabb(obj_copy.get_aabb(), obj_aabb) < max_dist)
            {
                Vector3f p_obj_0;
                if (old_dist_struct.is_null)
                    p_obj_0 = obj.htm.block(0, 0, 3, 1);
                else
                {
                    try
                    {
                        DistStructLinkObj dslo = old_dist_struct.get_item(ind_links, ind_obj_link);
                        if (!dslo.is_null)
                            p_obj_0 = dslo.point_object;
                        else
                            p_obj_0 = obj.htm.block(0, 0, 3, 1);
                    }
                    catch (const std::exception &e)
                    {
                        p_obj_0 = obj.htm.block(0, 0, 3, 1);
                    }
                }

                PrimDistResult pdr = obj.dist_to(obj_copy, h, eps, tol, no_iter_max, p_obj_0);

                DistStructLinkObj dslo_new;

                dslo_new.is_null = false;
                dslo_new.distance = pdr.dist;
                dslo_new.link_number = ind_links;
                dslo_new.link_col_obj_number = ind_obj_link;
                dslo_new.point_object = pdr.proj_A;
                dslo_new.point_link = pdr.proj_B;
                MatrixXf dv = (pdr.proj_B - pdr.proj_A).transpose();
                dslo_new.jac_distance = (dv * Jvv - (dv * s_mat(pdr.proj_A)) * Jw) / (pdr.dist + 1e-6);

                if (obj.type == 3 && !(h < 1e-5 && eps < 1e-5))
                {
                    dslo_new.jac_distance += (pdr.proj_B.cross(pdr.proj_A) - pdr.aux).transpose() * Jw / (pdr.dist + 1e-6);
                    // cout << "aaa = " << ((pdr.proj_B.cross(pdr.proj_A) - pdr.aux).transpose() * Jw / (pdr.dist + 1e-6)).norm() << std::endl;
                }

                old_rows = jac_tot.rows();
                jac_tot.conservativeResize(old_rows + 1, Eigen::NoChange);
                jac_tot.block(old_rows, 0, 1, dslo_new.jac_distance.cols()) = dslo_new.jac_distance;

                old_rows = dist_tot.rows();
                dist_tot.conservativeResize(old_rows + 1, Eigen::NoChange);
                dist_tot[old_rows] = pdr.dist;

                dsro.list_info.push_back(dslo_new);
            }
        }
    }

    dsro.jac_dist_mat = jac_tot;
    dsro.dist_vect = dist_tot;

    return dsro;
}

DistStructLinkLink::DistStructLinkLink() {};

DistStructLinkLink DistStructRobotAuto::get_item(int ind_link_1, int ind_link_2, int ind_obj_link_1, int ind_obj_link_2)
{
    for (int ind_info = 0; ind_info < list_info.size(); ind_info++)
    {
        if ((list_info[ind_info].link_number_1 == ind_link_1) &&
            (list_info[ind_info].link_number_2 == ind_link_2) &&
            (list_info[ind_info].link_col_obj_number_1 == ind_obj_link_1) &&
            (list_info[ind_info].link_col_obj_number_2 == ind_obj_link_2))
            return list_info[ind_info];
    }

    DistStructLinkLink dsll;
    dsll.is_null = true;

    return dsll;
}

DistStructRobotAuto::DistStructRobotAuto() {};

DistStructRobotAuto Manipulator::compute_dist_auto(VectorXf q, DistStructRobotAuto old_dist_struct,
                                                   float tol, int no_iter_max, float max_dist, float h, float eps) const
{
    FKResult fkres = fk(q, this->htm_world_to_dh0, true);

    DistStructRobotAuto dsra;
    dsra.is_null = false;
    dsra.list_info = {};

    MatrixXf jac_tot = MatrixXf::Zero(0, q.rows());
    VectorXf dist_tot = VectorXf::Zero(0);
    int old_rows;

    vector<MatrixXf> Jvv = vector<MatrixXf>();

    for (int ind_links = 0; ind_links < no_links; ind_links++)
    {
        MatrixXf Jv = fkres.jac_v_dh[ind_links];
        MatrixXf Jw = fkres.jac_w_dh[ind_links];
        Vector3f pc = fkres.get_p_dh(ind_links);
        Jvv.push_back(Jv + s_mat(pc) * Jw);
    }

    for (int ind_links_1 = 0; ind_links_1 < no_links; ind_links_1++)
    {
        MatrixXf Jv1 = fkres.jac_v_dh[ind_links_1];
        MatrixXf Jw1 = fkres.jac_w_dh[ind_links_1];
        Vector3f pc1 = fkres.get_p_dh(ind_links_1);
        MatrixXf Jvv1 = Jvv[ind_links_1];

        for (int ind_links_2 = ind_links_1 + 2; ind_links_2 < no_links; ind_links_2++)
        {
            MatrixXf Jv2 = fkres.jac_v_dh[ind_links_2];
            MatrixXf Jw2 = fkres.jac_w_dh[ind_links_2];
            Vector3f pc2 = fkres.get_p_dh(ind_links_2);
            MatrixXf Jvv2 = Jvv[ind_links_2];

            for (int ind_obj_link_1 = 0; ind_obj_link_1 < geo_prim[ind_links_1].size(); ind_obj_link_1++)
                for (int ind_obj_link_2 = 0; ind_obj_link_2 < geo_prim[ind_links_2].size(); ind_obj_link_2++)
                {

                    GeometricPrimitives obj_copy_1 = geo_prim[ind_links_1][ind_obj_link_1].copy();
                    obj_copy_1.htm = fkres.htm_dh[ind_links_1] * geo_prim[ind_links_1][ind_obj_link_1].htm;

                    GeometricPrimitives obj_copy_2 = geo_prim[ind_links_2][ind_obj_link_2].copy();
                    obj_copy_2.htm = fkres.htm_dh[ind_links_2] * geo_prim[ind_links_2][ind_obj_link_2].htm;

                    if (AABB::dist_aabb(obj_copy_1.get_aabb(), obj_copy_2.get_aabb()) < max_dist)
                    {
                        Vector3f p_obj_1;
                        if (old_dist_struct.is_null)
                            p_obj_1 = obj_copy_1.htm.block(0, 0, 3, 1);
                        else
                        {
                            try
                            {
                                DistStructLinkLink dsll = old_dist_struct.get_item(ind_links_1, ind_links_2, ind_obj_link_1, ind_obj_link_2);
                                if (!dsll.is_null)
                                    p_obj_1 = dsll.point_link_1;
                                else
                                    p_obj_1 = obj_copy_1.htm.block(0, 0, 3, 1);
                            }
                            catch (const std::exception &e)
                            {
                                p_obj_1 = obj_copy_1.htm.block(0, 0, 3, 1);
                            }
                        }

                        PrimDistResult pdr = obj_copy_1.dist_to(obj_copy_2, h, eps, tol, no_iter_max, p_obj_1);

                        DistStructLinkLink dsll_new;

                        dsll_new.is_null = false;
                        dsll_new.distance = pdr.dist;
                        dsll_new.link_number_1 = ind_links_1;
                        dsll_new.link_number_2 = ind_links_2;
                        dsll_new.link_col_obj_number_1 = ind_obj_link_1;
                        dsll_new.link_col_obj_number_2 = ind_obj_link_2;
                        dsll_new.point_link_1 = pdr.proj_A;
                        dsll_new.point_link_2 = pdr.proj_B;

                        MatrixXf dv = (pdr.proj_A - pdr.proj_B).transpose();
                        dsll_new.jac_distance = (dv * Jvv1 - (dv * s_mat(pdr.proj_A)) * Jw1) / (pdr.dist + 1e-6);
                        dsll_new.jac_distance += -(dv * Jvv2 - (dv * s_mat(pdr.proj_B)) * Jw2) / (pdr.dist + 1e-6);

                        old_rows = jac_tot.rows();
                        jac_tot.conservativeResize(old_rows + 1, Eigen::NoChange);
                        jac_tot.block(old_rows, 0, 1, dsll_new.jac_distance.cols()) = dsll_new.jac_distance;

                        old_rows = dist_tot.rows();
                        dist_tot.conservativeResize(old_rows + 1, Eigen::NoChange);
                        dist_tot[old_rows] = pdr.dist;

                        dsra.list_info.push_back(dsll_new);
                    }
                }
        }
    }

    dsra.jac_dist_mat = jac_tot;
    dsra.dist_vect = dist_tot;

    return dsra;
}

// -----------------------------------------------------------------------------
// ------------------------- VECTOR FIELD ON SE(3) -----------------------------
// -----------------------------------------------------------------------------

// Global constants for the default delta and ds values
double c_delta = 0.001;
double c_ds = 0.001;
// Approximate value for theta=0 in SE(3) EE distance
double c_maxCosTheta = 0.999;
// Upper bound for theta approximately zero for se3::exp
double c_theta_zero = 1e-6;

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, double, double, double, double> EEdistSE3Variables(const Eigen::Matrix4d &X)
{
    // Compute the variables used in the explicit EEdistance function in SE(3)
    Eigen::MatrixXd Z = X;
    Eigen::Matrix3d Q = Z.block<3, 3>(0, 0);
    Eigen::Vector3d t = Z.block<3, 1>(0, 3);
    double cos_theta = 0.5 * (Q.trace() - 1);
    double sin_theta = 1.0 / (2.0 * sqrt(2)) * (Q - Q.inverse()).norm();
    double theta = atan2(sin_theta, cos_theta);
    cos_theta = cos(theta);
    sin_theta = sin(theta);

    double alpha;
    if (cos_theta > c_maxCosTheta)
    {
        alpha = -(1.0 / 12);
    }
    else
    {
        alpha =
            (2.0 - 2 * cos_theta - pow(theta, 2)) / (4.0 * pow((1 - cos_theta), 2));
    }
    Eigen::Matrix3d M = alpha * (Q + Q.inverse()) +
                        (1 - 2 * alpha) * Eigen::Matrix3d::Identity();
    return std::make_tuple(Q, t, M, theta, cos_theta, sin_theta, alpha);
}

double EEdistance(const Eigen::Matrix4d V, const Eigen::Matrix4d W)
{
    Eigen::Matrix4d Z = V.inverse() * W;
    // auto [Q, t, M, theta, cos_theta, sin_theta, alpha] = EEdistSE3Variables(Z);
    std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, double, double, double, double> res_ = EEdistSE3Variables(Z);
    Eigen::Matrix3d Q;
    Eigen::Vector3d t;
    Eigen::Matrix3d M;
    double theta;
    double cos_theta;
    double sin_theta;
    double alpha;
    std::tie(Q, t, M, theta, cos_theta, sin_theta, alpha) = res_;

    double distance = sqrt(2.0 * pow(theta, 2) + t.transpose() * M * t);
    return distance;
}

std::tuple<double, int> ECdistance(const Eigen::MatrixXd &state, const vector<Eigen::Matrix4d> &curve)
{
    Eigen::Matrix4d V = state;
    int ind_min = 0;
    double min_distance = 1e6;
    for (int i = 0; i < curve.size(); i++)
    {
        double distance = EEdistance(state, curve.at(i));
        if (distance < min_distance)
        {
            min_distance = distance;
            ind_min = i;
        }
    }
    return {min_distance, ind_min};
}

Eigen::Matrix4d SmapSE3(const Eigen::VectorXd xi)
{
    Eigen::MatrixXd S_ = Eigen::Matrix4d::Zero();

    // Portion of Lie algebra related to SO(3)
    S_(0, 1) = -xi(5);
    S_(0, 2) = xi(4);
    S_(1, 2) = -xi(3);
    S_ = S_ - S_.transpose().eval();

    S_(0, 3) = xi(0);
    S_(1, 3) = xi(1);
    S_(2, 3) = xi(2);

    return S_;
}

Eigen::VectorXd invSmapSE3(const Eigen::Matrix4d A)
{
    Eigen::VectorXd xi = Eigen::VectorXd::Zero(6);

    xi(0) = A(0, 3);
    xi(1) = A(1, 3);
    xi(2) = A(2, 3);
    xi(3) = -A(1, 2);
    xi(4) = A(0, 2);
    xi(5) = -A(0, 1);

    return xi;
}

Eigen::Matrix4d expSE3(const Eigen::Matrix4d X)
{
    Eigen::Matrix3d A = X.block<3, 3>(0, 0);
    Eigen::Vector3d v = X.block<3, 1>(0, 3);
    Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
    float theta = sqrt(pow(A(1, 0), 2) + pow(A(0, 2), 2) + pow(A(2, 1), 2));
    // If theta is close to zero, use the first order approximation
    if (theta < c_theta_zero)
    {
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        result.block<3, 3>(0, 0) = R;
        result.block<3, 1>(0, 3) = v;
    }
    else
    {
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + (sin(theta) / theta) * A +
                            ((1 - cos(theta)) / pow(theta, 2)) * A * A;
        Eigen::Matrix3d U = Eigen::Matrix3d::Identity() +
                            ((1 - cos(theta)) / pow(theta, 2)) * A +
                            ((theta - sin(theta)) / pow(theta, 3)) * A * A;
        result.block<3, 3>(0, 0) = R;
        result.block<3, 1>(0, 3) = U * v;
    }
    return result;
}


// TODO: Implement explicit analytic computation of the L operator (normal component)
VectorFieldResult vectorfield_SE3(const Eigen::Matrix4d &state, const vector<Eigen::Matrix4d> &curve, float kt1, float kt2, float kt3, float kn1, float kn2,
                                  const vector<Eigen::MatrixXd> &curve_derivative, double delta, double ds)
{
    // auto [min_distance, closest_index] = ECdistance(state, curve);
    std::tuple<double, int> res_ = ECdistance(state, curve);
    double min_distance;
    int closest_index;
    std::tie(min_distance, closest_index) = res_;

    int candidate1 = closest_index+1<curve.size()? closest_index+1:0;
    int candidate2 = closest_index-1>0? closest_index-1:curve.size()-1;

    int second_closest_index=0;

    if(EEdistance(curve.at(candidate1),state) < EEdistance(curve.at(candidate2),state))
        second_closest_index=candidate1;
    else
        second_closest_index=candidate2;
    

    Eigen::Matrix4d closest_point = curve.at(closest_index);
    Eigen::Matrix4d second_closest_point = curve.at(second_closest_index);
    float second_distance = EEdistance(curve.at(second_closest_index),state);

    // Normal component
    int m = 6;
    Eigen::VectorXd normal = Eigen::VectorXd::Zero(m);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(m, m);
    // L-operator wrt V of EEdistance
    Eigen::VectorXd LvDhat = Eigen::VectorXd::Zero(m);

    for (int i = 0; i < m; i++)
    {
        Eigen::MatrixXd variation = expSE3(SmapSE3(I.col(i)) * delta) * state;
        double dDistance = EEdistance(variation, closest_point);
        LvDhat(i) = (dDistance - min_distance) / (delta);
    }
    normal = -LvDhat;

    Eigen::VectorXd normal2 = Eigen::VectorXd::Zero(m);

    for (int i = 0; i < m; i++)
    {
        Eigen::MatrixXd variation = expSE3(SmapSE3(I.col(i)) * delta) * state;
        double dDistance = EEdistance(variation, second_closest_point);
        LvDhat(i) = (dDistance - second_distance) / (delta);
    }
    normal2 = -LvDhat;


    // Tangent Component
    Eigen::MatrixXd dHd;
    if (curve_derivative.size() > 0)
    {
        dHd = curve_derivative.at(closest_index);
    }
    else
    {
        if (closest_index == curve.size() - 1)
        {
            // If the closest point is the last point on the curve, the next point
            // is the first point (closed curve).
            dHd = (curve.at(0) - closest_point) / ds;
        }
        else
        {
            Eigen::MatrixXd next_point = curve.at(closest_index + 1);
            dHd = (next_point - closest_point) / ds;
        }
    }
    Eigen::VectorXd tangent = invSmapSE3(dHd * closest_point.inverse());


    if (curve_derivative.size() > 0)
    {
        dHd = curve_derivative.at(second_closest_index);
    }
    else
    {
        if (second_closest_index == curve.size() - 1)
        {
            // If the closest point is the last point on the curve, the next point
            // is the first point (closed curve).
            dHd = (curve.at(0) - second_closest_point) / ds;
        }
        else
        {
            Eigen::MatrixXd next_point = curve.at(second_closest_index + 1);
            dHd = (next_point - second_closest_point) / ds;
        }
    }
    Eigen::VectorXd tangent2 = invSmapSE3(dHd * second_closest_point.inverse());

    Matrix4d A = state - closest_point;
    Matrix4d B = second_closest_point - closest_point;
    float alpha = A.cwiseProduct(B).sum() / B.squaredNorm();
    alpha = minf(maxf(alpha,0),1);

    normal = (1-alpha)*normal+alpha*normal2;
    tangent = (1-alpha)*tangent+alpha*tangent2;


    double eps = 1e-3;

    double dist_mod = sqrt(min_distance + pow(eps, 2)) - eps;

    double kn = kn1 * std::tanh(kn2 * dist_mod);
    double kt = kt1 * (1 - kt2 * std::tanh(kt3 * dist_mod));

    // double kt = kt1 * (1 - kt2 * std::tanh(kt3 * min_distance));
    // double kn = kn1 * std::tanh(kn2 * min_distance);

    VectorFieldResult vfr;
    Eigen::VectorXd twist = kt * tangent + kn * normal;
    vfr.twist = twist.cast<float>();
    vfr.dist = min_distance;
    vfr.index = closest_index;
    vfr.normal = (kn * normal).cast<float>();
    vfr.tangent = (kt * tangent).cast<float>();

    return vfr;
}

// -----------------------------------------------------------------------------
// ------------------------- VECTOR FIELD ON SE(3) END -------------------------
// -----------------------------------------------------------------------------

#include <math.h>

#define PI           3.1415926
#define SQRTHALFPI   1.2533141
#define SQRT2        1.4142135
#define CONSTJA      2.7889

static inline double J(double u) {
    double pu2 = PI * u * u;
    return CONSTJA / (((CONSTJA - 1.0) * sqrt(pu2)) + sqrt(pu2 + CONSTJA * CONSTJA));
}

double Int(double v, double h, double L) {
    if (fabs(v) <= L) {
        double A1 = exp(-((L - v) * (L - v)) / (2.0 * h * h)) * J((v - L) / (SQRT2 * h));
        double A2 = exp(-((L + v) * (L + v)) / (2.0 * h * h)) * J((v + L) / (SQRT2 * h));
        return -h * h * log(SQRTHALFPI * (h / (2.0 * L)) * (2.0 - A1 - A2));
    } else {
        // The function is even
        v = fabs(v);
        double A1 = J((v - L) / (SQRT2 * h));
        double A2 = exp(-2.0 * L * v / (h * h)) * J((v + L) / (SQRT2 * h));
        return 0.5 * (v - L) * (v - L) - h * h * log(SQRTHALFPI * (h / (2.0 * L)) * (A1 - A2));
    }
}


#include <math.h>

#define PI           3.1415926
#define SQRTHALFPI   1.2533141
#define SQRT2        1.4142135

static inline double I0_hat(double u) {
    /* pow(1 + 0.25*u*u, -0.25) * (1 + 0.24273*u*u) / (1 + 0.43023*u*u) */
    double u2 = u * u;
    return pow(1.0 + 0.25 * u2, -0.25) * (1.0 + 0.24273 * u2) / (1.0 + 0.43023 * u2);
}

static inline double f(double nu, double rho) {
    double d1 = rho - nu;
    double d2 = rho + nu;
    double A1 = exp(-0.5 * d1 * d1);
    double A2 = exp(-0.5 * d2 * d2);
    return rho * (A1 + A2) * I0_hat(rho * nu);
}

static inline double f_hat(double nu, double rho, double rhobar) {
    double d1 = rho - nu;
    double d2 = rho + nu;
    double shift = 0.5 * (rhobar - nu) * (rhobar - nu);
    double A1 = exp(-0.5 * d1 * d1 + shift);
    double A2 = exp(-0.5 * d2 * d2 + shift);
    return rho * (A1 + A2) * I0_hat(rho * nu);
}

/* Avoid potential clashes with macros by naming these dmax/dmin */
static inline double dmax(double a, double b) { return (a >= b) ? a : b; }
static inline double dmin(double a, double b) { return (a <= b) ? a : b; }

double Cir(double v, double h, double R) {
    /* The function should be called only for v >= 0; the integrand is even */
    v = fabs(v);

    /* 7-point Gauss–Legendre quadrature */
    const int N = 7;
    static const double node[7]   = {-0.94910, -0.74153, -0.40584, 0.0, 0.40584, 0.74153, 0.94910};
    static const double weight[7] = { 0.12948,  0.27970,  0.38183, 0.41790, 0.38183, 0.27970, 0.12948};

    double Flow, Fup, delta, rhobar, y;

    if (v <= R) {
        Flow  = dmax(0.0, sqrt((v / h) * (v / h) + 1.0) - 3.0);
        Fup   = dmin(R / h, sqrt((v / h) * (v / h) + 1.0) + 3.0);
        delta = 0.5 * (Fup - Flow);

        y = 0.0;
        for (int i = 0; i < N; ++i) {
            double rho = Flow + delta * (node[i] + 1.0);
            y += weight[i] * f(v / h, rho);
        }
        y *= delta;

        return -h * h * log(y * (h / R) * (h / R));
    } else {
        Flow  = 0.0;
        Fup   = R / h;
        delta = 0.5 * (Fup - Flow);

        rhobar = Flow + delta * (node[N - 1] + 1.0);

        y = 0.0;
        for (int i = 0; i < N; ++i) {
            double rho = Flow + delta * (node[i] + 1.0);
            y += weight[i] * f_hat(v / h, rho, rhobar);
        }
        y *= delta;

        return 0.5 * (v - h * rhobar) * (v - h * rhobar)
             - h * h * log(y * (h / R) * (h / R));
    }
}


float gon_E(Vector3f point, GeometricPrimitives obj, float h)
{
    Vector3f pc = obj.htm.block<3, 1>(0, 3);
    Matrix3f Q = obj.htm.block<3, 3>(0, 0);
    Vector3f point_transformed = Q.transpose() * (point - pc);
    float x = point_transformed[0];
    float y = point_transformed[1];
    float z = point_transformed[2];

    if(obj.type==1)
        return Int(x,h,obj.lx/2)+Int(y,h,obj.ly/2)+Int(z,h,obj.lz/2);
    if(obj.type==2)
        return Cir(sqrtf(x*x+y*y),h,obj.lx)+Int(z,h,obj.lz/2);
}

Vector3f gon_proj(Vector3f point, GeometricPrimitives obj, float h)
{
    Vector3f proj;
    float eps=0.01;
    Vector3f dx, dy, dz;
    dx << eps, 0, 0;
    dy << 0, eps, 0;
    dz << 0, 0, eps;

    // proj[0] = point[0] - (gon_E(point+dx,obj,h)-gon_E(point-dx,obj,h))/(2*eps);
    // proj[1] = point[1] - (gon_E(point+dy,obj,h)-gon_E(point-dy,obj,h))/(2*eps);
    // proj[2] = point[2] - (gon_E(point+dz,obj,h)-gon_E(point-dz,obj,h))/(2*eps);

    float E = gon_E(point,obj,h);
    proj[0] = point[0] - (gon_E(point+dx,obj,h)-E)/(eps);
    proj[1] = point[1] - (gon_E(point+dy,obj,h)-E)/(eps);
    proj[2] = point[2] - (gon_E(point+dz,obj,h)-E)/(eps);


    return proj;

}

PrimDistResult dist_to_gon(GeometricPrimitives prim_a, GeometricPrimitives prim_b, float h, float tol, Vector3f p_A0, int no_iter_max) 
{
    Vector3f a, b, a_next;

    a = p_A0;

    bool cont = true;

    vector<float> hist_error = {};
    int iter = 0;
    while(cont)
    {
        b = gon_proj(a, prim_b, h);
        a_next = gon_proj(b, prim_a, h);
        float error = (a-a_next).norm();
        cont = (error >= tol) && (iter <= no_iter_max);
        hist_error.push_back(error);
        a = a_next;
        iter++;
    }

    PrimDistResult pdr;
    pdr.hist_error = hist_error;
    pdr.proj_A = a;
    pdr.proj_B = b;

    return pdr;


}

//////////


// Skew-symmetric matrix S(omega)
inline Eigen::Matrix3f skew(const Eigen::Vector3f& w) {
    Eigen::Matrix3f S;
    S <<     0, -w(2),  w(1),
          w(2),     0, -w(0),
         -w(1),  w(0),     0;
    return S;
}

// Saturation: clamp each element between tau_min and tau_max
inline Eigen::VectorXf saturate(const Eigen::VectorXf& v, float tau_min, float tau_max) {
    Eigen::VectorXf out = v;
    for (int i = 0; i < v.size(); i++) {
        out(i) = std::min(std::max(v(i), tau_min), tau_max);
    }
    return out;
}


//////////////////////////





static inline float smooth_min_h(const std::vector<float>& s, float h, float eps)
{
    float min_s = std::numeric_limits<float>::infinity();
    for (float v : s) min_s = std::min(min_s, v);
    min_s = std::max(min_s, eps);

    // Compute S = sum (min_s/s_k)^h
    double S = 0.0;
    for (float v : s)
    {
        float sv = std::max(v, eps);
        double a = std::pow(double(min_s) / double(sv), double(h));
        S += a;
    }

    // min_s * S^(-1/h)
    double out = double(min_s) * std::pow(S, -1.0 / double(h));
    return float(out);
}

static inline std::vector<float> smooth_min_h_grad_wrt_inputs(
    const std::vector<float>& s, float h, float eps, float m_out)
{
    const int n = (int)s.size();
    std::vector<float> w(n, 0.0f);

    float min_s = std::numeric_limits<float>::infinity();
    for (float v : s) min_s = std::min(min_s, v);
    min_s = std::max(min_s, eps);

    // Compute a_k and S
    std::vector<double> a(n, 0.0);
    double S = 0.0;
    for (int k = 0; k < n; ++k)
    {
        float sk = std::max(s[k], eps);
        a[k] = std::pow(double(min_s) / double(sk), double(h));
        S += a[k];
    }

    // w_k = d m / d s_k
    for (int k = 0; k < n; ++k)
    {
        float sk = std::max(s[k], eps);
        double wk = double(m_out) * a[k] / (S * double(sk));
        w[k] = float(wk);
    }
    return w;
}

static inline float smooth_max_h_and_weights(
    const std::vector<float>& s, float h, float eps, std::vector<float>& dmax_ds)
{
    const int n = (int)s.size();
    dmax_ds.assign(n, 0.0f);

    // t_k = 1/s_k
    std::vector<float> t(n);
    for (int k = 0; k < n; ++k)
    {
        float sk = std::max(s[k], eps);
        t[k] = 1.0f / sk;
    }

    float m_t = smooth_min_h(t, h, eps);
    float max_out = 1.0f / std::max(m_t, eps);

    // For dmax/ds, we need normalized weights on t:
    // Let a_k=(min_t/t_k)^h, S=sum a_k, then normalized u_k=a_k/S.
    // Then dmax/ds_k = max_out * u_k / s_k
    float min_t = std::numeric_limits<float>::infinity();
    for (float v : t) min_t = std::min(min_t, v);
    min_t = std::max(min_t, eps);

    std::vector<double> a(n, 0.0);
    double S = 0.0;
    for (int k = 0; k < n; ++k)
    {
        float tk = std::max(t[k], eps);
        a[k] = std::pow(double(min_t) / double(tk), double(h));
        S += a[k];
    }

    for (int k = 0; k < n; ++k)
    {
        float sk = std::max(s[k], eps);
        double u_k = a[k] / S;                  // normalized
        double wk = double(max_out) * u_k / double(sk);
        dmax_ds[k] = float(wk);
    }

    return max_out;
}

// ICUASGVF icuas_gvf(
//     const std::vector<Eigen::Vector3f>& q,
//     const std::vector<std::vector<Eigen::Vector3f>>& q_tg,
//     float h = 10.0f,
//     float Kc = 1.0,
//     float Kt = 1.0,
//     float eps = 1e-8f)
// {
//     const int J = (int)q.size();
//     const int I = (int)q_tg.size();

//     if (I == 0 || J == 0)
//         throw std::runtime_error("compute_comp_icuas: empty inputs.");

//     for (int i = 0; i < I; ++i)
//     {
//         if ((int)q_tg[i].size() != J)
//             throw std::runtime_error("compute_comp_icuas: each q_tg[i] must have size q.size().");
//     }

//     // 1) Compute d_ij = 0.5 ||q[j]-q_tg[i][j]||^2 (clamped to eps for stability)
//     std::vector<std::vector<float>> d(I, std::vector<float>(J, 0.0f));
//     for (int i = 0; i < I; ++i)
//     {
//         for (int j = 0; j < J; ++j)
//         {
//             Eigen::Vector3f diff = q[j] - q_tg[i][j];
//             float dij = 0.5f * diff.squaredNorm();
//             d[i][j] = std::max(dij, eps);
//         }
//     }

//     // 2) Inner smooth max over j: M_i = max_h_j d_ij, and beta_ij = dM_i/dd_ij
//     std::vector<float> M(I, 0.0f);
//     std::vector<std::vector<float>> beta(I, std::vector<float>(J, 0.0f));

//     for (int i = 0; i < I; ++i)
//     {
//         std::vector<float> dmax_dd; // size J: dM/dd_ij
//         M[i] = smooth_max_h_and_weights(d[i], h, eps, dmax_dd);
//         for (int j = 0; j < J; ++j)
//             beta[i][j] = dmax_dd[j];
//     }

//     // 3) Outer smooth min over i: D = min_h_i M_i, alpha_i = dD/dM_i
//     float D = smooth_min_h(M, 20, eps);
//     std::vector<float> alpha = smooth_min_h_grad_wrt_inputs(M, 20, eps, D);

//     // 4) Combine weights: gamma_ij = dD/dd_ij = alpha_i * beta_ij
//     //    grad_D[j] = sum_i gamma_ij * (q[j]-q_tg[i][j])
//     //    T[j]      = sum_i (gamma_ij * I3) * T_tg[i][j] = sum_i gamma_ij * T_tg[i][j]
//     std::vector<Eigen::Vector3f> vel;

//     vel.assign(J, Eigen::Vector3f::Zero());


//     float i_approx = 0;
//     float sum_alpha=0;

//     for (int i = 0; i < I; ++i)
//     {
//         i_approx+=alpha[i]*i;
//         sum_alpha+=alpha[i];
//     }

//     i_approx=i_approx/sum_alpha;

//     int ii_approx = static_cast<int>(std::floor(i_approx));

//     float norm = 0;

//     for (int j = 0; j < J; ++j)
//         norm+=(q_tg[(ii_approx+1)%I][j]-q_tg[(ii_approx-1+I)%I][j]).norm();

//     for (int j = 0; j < J; ++j)
//     {
//         Eigen::Vector3f g = Eigen::Vector3f::Zero();
//         Eigen::Vector3f t = Eigen::Vector3f::Zero();

//         float gamma_sum = 0;

//         for (int i = 0; i < I; ++i)
//         {
//             float gamma = alpha[i] * beta[i][j];  // scalar

//             gamma_sum+=gamma;
  
//             // W[i][j] = gamma * I3 (implicit)
//             g += gamma * (q[j] - q_tg[i][j]);
//         }

//         g = g/(2*sqrtf(D)+eps);

//         float G = (2/3.15)*atan(Kc*sqrtf(D));
//         float H = Kt * sqrt(1.0+eps-G*G);
//         t = (q_tg[(ii_approx+1)%I][j]-q_tg[(ii_approx-1+I)%I][j])/norm;

//         // cout<<"----"<<std::endl;
//         // cout << G<<std::endl;
//         // cout << H<<std::endl;
//         // cout<<t[0]<<t[1]<<t[2]<<std::endl;

//         vel[j] = -G*g+H*t;

//     }

//     ICUASGVF out;
//     out.D = D;
//     out.vel = vel;


//     return out;
// }



ICUASGVF icuas_gvf_vel(
    const std::vector<Eigen::Vector3f>& q,
    const std::vector<std::vector<Eigen::Vector3f>>& q_tg,
    float h = 10.0f,
    float Kc = 1.0,
    float Kt = 1.0,
    float eps = 1e-8f)
{
    const int J = (int)q.size();
    const int I = (int)q_tg.size();

    if (I == 0 || J == 0)
        throw std::runtime_error("compute_comp_icuas: empty inputs.");

    for (int i = 0; i < I; ++i)
    {
        if ((int)q_tg[i].size() != J)
            throw std::runtime_error("compute_comp_icuas: each q_tg[i] must have size q.size().");
    }

    // 1) Compute d_ij = 0.5 ||q[j]-q_tg[i][j]||^2 (clamped to eps for stability)
    std::vector<std::vector<float>> d(I, std::vector<float>(J, 0.0f));
    for (int i = 0; i < I; ++i)
    {
        for (int j = 0; j < J; ++j)
        {
            Eigen::Vector3f diff = q[j] - q_tg[i][j];
            float dij = 0.5f * diff.squaredNorm();
            d[i][j] = std::max(dij, eps);
        }
    }

    // 2) Inner smooth max over j: M_i = max_h_j d_ij, and beta_ij = dM_i/dd_ij
    std::vector<float> M(I, 0.0f);
    std::vector<std::vector<float>> beta(I, std::vector<float>(J, 0.0f));

    for (int i = 0; i < I; ++i)
    {
        std::vector<float> dmax_dd; // size J: dM/dd_ij
        M[i] = smooth_max_h_and_weights(d[i], h, eps, dmax_dd);
        for (int j = 0; j < J; ++j)
            beta[i][j] = dmax_dd[j];
    }

    // 3) Outer smooth min over i: D = min_h_i M_i, alpha_i = dD/dM_i
    float D = smooth_min_h(M, 80, eps);
    std::vector<float> alpha = smooth_min_h_grad_wrt_inputs(M, 80, eps, D);

    // 4) Combine weights: gamma_ij = dD/dd_ij = alpha_i * beta_ij
    //    grad_D[j] = sum_i gamma_ij * (q[j]-q_tg[i][j])
    //    T[j]      = sum_i (gamma_ij * I3) * T_tg[i][j] = sum_i gamma_ij * T_tg[i][j]
    std::vector<Eigen::Vector3f> vel;

    vel.assign(J, Eigen::Vector3f::Zero());


    float i_approx = 0;
    float sum_alpha=0;

    for (int i = 0; i < I; ++i)
    {
        i_approx+=alpha[i]*i;
        sum_alpha+=alpha[i];
    }

    i_approx=i_approx/sum_alpha;

    int ii_approx = static_cast<int>(std::floor(i_approx));

    //cout<<"ii_approx = "<<ii_approx<<std::endl;

    float norm = 0;

    for (int j = 0; j < J; ++j)
        norm+=(q_tg[(ii_approx+1)%I][j]-q_tg[(ii_approx-1+I)%I][j]).norm();

    for (int j = 0; j < J; ++j)
    {
        Eigen::Vector3f g = Eigen::Vector3f::Zero();
        Eigen::Vector3f t = Eigen::Vector3f::Zero();

        float gamma_sum = 0;

        for (int i = 0; i < I; ++i)
        {
            float gamma = alpha[i] * beta[i][j];  // scalar

            gamma_sum+=gamma;
  
            // W[i][j] = gamma * I3 (implicit)
            g += gamma * (q[j] - q_tg[i][j]);
        }

        g = g/(2*sqrtf(D)+eps);

        float G = (2/3.15)*atan(Kc*sqrtf(D));
        float H = Kt * sqrt(1.0+eps-G*G);
        t = (q_tg[(ii_approx+1)%I][j]-q_tg[(ii_approx-1+I)%I][j])/norm;

        // cout<<"----"<<std::endl;
        // cout << G<<std::endl;
        // cout << H<<std::endl;
        //cout<<j<<": "<<t[0]<<","<<t[1]<<","<<t[2]<<std::endl;

        vel[j] = -G*g+H*t;

    }

    ICUASGVF out;
    out.D = D;
    out.vec = vel;


    return out;
}

ICUASGVF icuas_gvf_acc(
    const std::vector<Eigen::Vector3f>& q,
    const std::vector<Eigen::Vector3f>& dotq,
    const std::vector<GeometricPrimitives>& obstacles,
    const std::vector<std::vector<Eigen::Vector3f>>& moving_obstacles,
    const std::vector<std::vector<Eigen::Vector3f>>& q_tg,
    float h = 10.0f,
    float Kc = 1.0,
    float Kt = 1.0,
    float tau_v = 0.2,
    float agent_radius=0.05,
    float agent_height=0.2,
    float r=3.0,
    float eps_d=0.01,
    float safe_dist = 1.5,
    float eta = 0.5,
    float dist_delta=0.03,
    float moving_obstacle_radius = 0.3,
    float eps = 1e-8f)
{


    float delta=0.05;

    const int J = (int)q.size();
    const int K = (int)obstacles.size();

    vector<Vector3f> q_next;
    q_next.assign(J, Eigen::Vector3f::Zero());

    for (int j=0; j<J;j++)
        q_next[j] = q[j]+delta*dotq[j];


    ICUASGVF vel = icuas_gvf_vel(q,q_tg,h,Kc,Kt,eps);
    ICUASGVF vel_next = icuas_gvf_vel(q_next,q_tg,h,Kc,Kt,eps);

    vector<Vector3f> a;
    a.assign(J, Eigen::Vector3f::Zero());

    for (int j=0; j<J;j++)
        a[j] = (vel_next.vec[j]-vel.vec[j])/delta + (vel.vec[j]-dotq[j])/tau_v;
    
    ICUASGVF out;

    //
    //cout<<"a"<<std::endl;

    vector<GeometricPrimitives> all_geo(J);
    vector<GeometricPrimitives> all_geo_next(J);

    for(int j=0; j<J; j++)
    {
        Matrix4f htm = trn(q[j][0],q[j][1],q[j][2]);
        Matrix4f htm_next = trn(q[j][0]+delta*dotq[j][0],q[j][1]+delta*dotq[j][1],q[j][2]+delta*dotq[j][2]);

        all_geo[j] = GeometricPrimitives::create_cylinder(htm, agent_radius, agent_height);
        all_geo_next[j] = GeometricPrimitives::create_cylinder(htm_next, agent_radius, agent_height);
    }

    //cout<<"b"<<std::endl;

    float min_dist_obs = 10000;
    float dist_temp;

    vector<float> dist_aa = {};
    vector<float> dist_aa_next = {};
    vector<Vector3f> grad_aa = {};
    vector<Vector3f> grad_aa_next = {};
    vector<int> ind1_aa = {};
    vector<int> ind2_aa = {};

    int k = 0;

    for(int j1 = 0; j1<J; j1++)
        for(int j2 = 0; j2<j1; j2++)
        {
            
            PrimDistResult res = all_geo[j1].dist_to(all_geo[j2],r,eps_d,1e-7,2000);
            PrimDistResult res_next = all_geo_next[j1].dist_to(all_geo_next[j2],r,eps_d,1e-7,2000);

            min_dist_obs = min(min_dist_obs, all_geo[j1].dist_to(all_geo[j2],1e-9,1e-9,1e-7,2000).dist);

            dist_aa.push_back(res.dist);
            dist_aa_next.push_back(res_next.dist);
            grad_aa.push_back((res.proj_A-res.proj_B).normalized());
            grad_aa_next.push_back((res_next.proj_A-res_next.proj_B).normalized());
            ind1_aa.push_back(j1);
            ind2_aa.push_back(j2);
            k++;
        }

    //cout<<"c"<<std::endl;

    //
    vector<float> dist_ao = {};
    vector<float> dist_ao_next = {};
    vector<Vector3f> grad_ao= {};
    vector<Vector3f> grad_ao_next= {};
    vector<int> ind1_ao = {};
    vector<int> ind2_ao = {};



    for(int j=0; j < J; j++)
        for(int k=0; k<K; k++)
        {
            dist_temp = all_geo[j].dist_to(obstacles[k], 1e-9, 1e-9, 1e-8, 2000).dist;
            min_dist_obs = min(min_dist_obs,dist_temp);

            if(dist_temp < safe_dist)
            {
                PrimDistResult res = all_geo[j].dist_to(obstacles[k],r,eps_d,1e-7,2000);
                PrimDistResult res_next = all_geo_next[j].dist_to(obstacles[k],r,eps_d,1e-7,2000);

                dist_ao.push_back(res.dist);
                dist_ao_next.push_back(res_next.dist);
                grad_ao.push_back((res.proj_A-res.proj_B).normalized());
                grad_ao_next.push_back((res_next.proj_A-res_next.proj_B).normalized());
                ind1_ao.push_back(j);
                ind2_ao.push_back(k);
            }
        }

    //
    //cout<<"c1"<<std::endl;

    vector<float> dist_am = {};
    vector<float> dist_am_next = {};
    vector<Vector3f> grad_am= {};
    vector<Vector3f> grad_am_next= {};
    vector<int> ind1_am = {};
    vector<int> ind2_am = {};

    for(int s=0; s<(int)moving_obstacles.size(); s++)
    {
        

        Vector3f pos = moving_obstacles[s][0];
        Vector3f vel = moving_obstacles[s][1];


        Matrix4f htm_mo = trn(pos[0],pos[1],pos[2]);
        Matrix4f htm_mo_next = trn(pos[0]+delta*vel[0],pos[1]+delta*vel[1],pos[2]+delta*vel[2]);

        GeometricPrimitives moving_obs = GeometricPrimitives::create_sphere(htm_mo, moving_obstacle_radius);
        GeometricPrimitives moving_obs_next = GeometricPrimitives::create_sphere(htm_mo_next, moving_obstacle_radius);

        for(int j=0; j < J; j++) 
        {
            dist_temp = all_geo[j].dist_to(moving_obs, 1e-9, 1e-9, 1e-8, 2000).dist;
            min_dist_obs = min(min_dist_obs,dist_temp);

            if(dist_temp < 2*safe_dist)
            {
                // cout<<"s = "<<s<<", j = "<<j<<std::endl;

                PrimDistResult res = all_geo[j].dist_to(moving_obs,r,eps_d,1e-7,2000);
                PrimDistResult res_next = all_geo_next[j].dist_to(moving_obs_next,r,eps_d,1e-7,2000);

                dist_am.push_back(res.dist);
                dist_am_next.push_back(res_next.dist);
                grad_am.push_back((res.proj_A-res.proj_B).normalized());
                grad_am_next.push_back((res_next.proj_A-res_next.proj_B).normalized());
                ind1_am.push_back(j);
                ind2_am.push_back(s);
            }
        }
    }
    //


    int N = (int)dist_aa.size();
    int M = (int)dist_ao.size();
    int S = (int)dist_am.size();

    //N=0;

    //cout<<"d"<<std::endl;

    MatrixXf A = MatrixXf::Zero(N+M+S+6*J, 3*J);
    VectorXf b = VectorXf::Zero(N+M+S+6*J);
    VectorXf f = VectorXf::Zero(3*J);
    MatrixXf H = MatrixXf::Identity(3*J, 3*J);

    for(int j=0; j<J; j++)
    {
        f[3*j+0] = -a[j][0];
        f[3*j+1] = -a[j][1];
        f[3*j+2] = -a[j][2];
    }

    for(int n=0; n < N; n++)
    {
        int i1 = ind1_aa[n];
        int i2 = ind2_aa[n];

        A(n,3*i1+0)= grad_aa[n][0];
        A(n,3*i1+1)= grad_aa[n][1];
        A(n,3*i1+2)= grad_aa[n][2];
        A(n,3*i2+0)=-grad_aa[n][0];
        A(n,3*i2+1)=-grad_aa[n][1];
        A(n,3*i2+2)=-grad_aa[n][2];

        float B = dist_aa[n]-dist_delta;
        float dBdt = grad_aa[n].dot(dotq[i1]-dotq[i2]);
        float ff = ((grad_aa_next[n]-grad_aa[n])/delta).dot(dotq[i1]-dotq[i2]);


        b[n] = -2*eta*dBdt-eta*eta*B-ff;
    }

    //cout<<"e"<<std::endl;

    for(int m=0; m < M; m++)
    {
        int i1 = ind1_ao[m];

        A(N+m,3*i1+0)= grad_ao[m][0];
        A(N+m,3*i1+1)= grad_ao[m][1];
        A(N+m,3*i1+2)= grad_ao[m][2];


        float B = dist_ao[m]-dist_delta;
        float dBdt = grad_ao[m].dot(dotq[i1]);
        float ff = ((grad_ao_next[m]-grad_ao[m])/delta).dot(dotq[i1]);


        b[N+m] = -2*eta*dBdt-eta*eta*B-ff;
    }

    int ind_special=-1;

    for(int s=0; s < S; s++)
    {
        int i1 = ind1_am[s];
        int i2 = ind2_am[s];

        A(N+M+s,3*i1+0)= grad_am[s][0];
        A(N+M+s,3*i1+1)= grad_am[s][1];
        A(N+M+s,3*i1+2)= grad_am[s][2];

        Vector3f vel = moving_obstacles[i2][1];
        Vector3f acc = moving_obstacles[i2][2];


        float B = dist_am[s]-7*dist_delta;
        float dBdt = grad_am[s].dot(dotq[i1]-vel);
        float ff = ((grad_am_next[s]-grad_am[s])/delta).dot(dotq[i1]-vel)-grad_am[s].dot(acc);
        ff = minf(ff,0);


        if(B>0)
            b[N+M+s] = -2*eta*dBdt-eta*eta*B-ff;
        else
            b[N+M+s] = -9*eta*eta*B-ff;
    }

    //
    for(int j=0; j < J; j++)
    {
        A(N+M+S+3*j+0,3*j+0)=1;
        A(N+M+S+3*j+1,3*j+1)=1;
        A(N+M+S+3*j+2,3*j+2)=1;
        A(N+M+S+3*J+3*j+0,3*j+0)=-1;
        A(N+M+S+3*J+3*j+1,3*j+1)=-1;
        A(N+M+S+3*J+3*j+2,3*j+2)=-1;


        b[N+M+S+3*j+0]=-2;
        b[N+M+S+3*j+1]=-2;
        b[N+M+S+3*j+2]=-2;
        b[N+M+S+3*J+3*j+0]=-2;
        b[N+M+S+3*J+3*j+1]=-2;
        b[N+M+S+3*J+3*j+2]=-2;
    }


    //Solve QP

    VectorXf af_aux = solveQP(H,f,A,b);
    vector<Vector3f> af = {};

    if(af_aux.rows()==0)
    {
        cout<<"Unfeasible!"<<std::endl;
        out.feasible=0;

        for(int j=0; j < J; j++)
            af.push_back(-eta*dotq[j]);
    
    }
    else
    {
        out.feasible=1;
        for(int j=0; j < J; j++)
        {
            Vector3f af_v;
            af_v[0]=af_aux[3*j+0];
            af_v[1]=af_aux[3*j+1];
            af_v[2]=af_aux[3*j+2];
            af.push_back(af_v);
        }


    }

    //

    out.dist_am = dist_am;
    out.dist_am_next = dist_am_next;
    out.grad_am= grad_am;
    out.grad_am_next= grad_am_next;
    out.ind1_am = ind1_am;
    out.ind2_am = ind2_am;


    //

    

    out.vec = af;
    out.D = sqrtf(2*vel.D);
    out.min_dist_obs = min_dist_obs;


    cout<<"----END-----"<<std::endl;

    return out;
}


/////



