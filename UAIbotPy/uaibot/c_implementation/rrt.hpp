#ifndef RRT_HPP
#define RRT_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <list>
#include <thread>
#include <mutex>
#include <atomic>
#include "gjk.h"
#include "nanoflann.hpp"
#include <queue>
#include <chrono>
#include <random>
#include <algorithm>

#include "declarations.h"

using namespace std;
using namespace Eigen;

#define SAMPLE_STEP 0.05f
#define INTERP_STEP 0.01f

//////////////////////////////////
// KDtree
/////////////////////////////////



struct RRTPointCloud {
    struct Point {
        std::vector<float> coords; 
    };
    std::vector<Point> pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return pts[idx].coords[dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTreeAdaptor = nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<float, RRTPointCloud>, RRTPointCloud, -1 >;

/////////////////////////////////

struct RRTResult
{
    bool success;
    vector<VectorXf> path;
    int iterations;
    float planning_time;

    RRTResult() : success(false), iterations(0), planning_time(0.0) {}

};

struct Node {
    Eigen::VectorXf q;   
    int parent_idx;  
    
    Node(const Eigen::VectorXf& q, int parent_idx) : q(q), parent_idx(parent_idx) {}
};

class RRT {
public:
    RRT(Manipulator robot,VectorXf q_start, vector<VectorXf> q_goal, Matrix4f htm, vector<GeometricPrimitives> obstacles,
        int max_iter, float goal_tolerance, float goal_bias, float step_size_min, float step_size_max, bool usemultthread
    );

    RRTResult runRRT();

private:

    // Auxiliary methods

    VectorXf sampleRandomConfig(); 

    VectorXf steer(const VectorXf& q_near, const VectorXf& q_rand); 

    bool isPathCollisionFree(const VectorXf& q1, const VectorXf& q2); 

    bool reachedGoal(const VectorXf& q_new);

    void addNode(const VectorXf& q_new, int parent_idx); 

    int getNearestNeighbor(const VectorXf& q_rand); 

    vector<VectorXf> backtrackPath(const int &goal_idx); 

    vector<VectorXf> shortcutting(const vector<VectorXf>& path);
    
    vector<VectorXf> interpolatePath(const vector<VectorXf>& path);

    bool verifyPath(const vector<VectorXf>& path);

    // Inputs    
    Manipulator robot;
    VectorXf q_start;
    vector<VectorXf> q_goal;
    Matrix4f htm_target;
    vector<GeometricPrimitives> obstacles;
    int max_iter;
    float goal_tolerance;
    float goal_bias;
    float step_size_min;
    float step_size_max;
    bool usemultthread;
    
    // Outputs
    RRTResult result;

    // internal variables
    vector<Node> tree;
    int n;
    VectorXf q_min;
    VectorXf q_max;
    Matrix4f htm;
    std::unique_ptr<KDTreeAdaptor> kdtree;
    std::unique_ptr<RRTPointCloud> pointcloud;
    std::mutex tree_mutex;
    std::atomic<bool> found;
};

#endif 