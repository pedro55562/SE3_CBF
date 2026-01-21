#ifndef SE3RRT_HPP
#define SE3RRT_HPP

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
#include "rrt.hpp"

using namespace std;
using namespace Eigen;

#define SAMPLE_STEP 0.05f
#define INTERP_STEP 0.01f


struct SE3Node {
    Eigen::VectorXf q;   
    int parent_idx;  
    
    SE3Node(const Eigen::VectorXf& q, int parent_idx) : q(q), parent_idx(parent_idx) {}
};

class SE3RRT {
public:
    SE3RRT(Manipulator robot,VectorXf q_start, vector<VectorXf> q_goal, Matrix4f htm, vector<GeometricPrimitives> obstacles,
        int max_iter, float goal_tolerance, float goal_bias, float step_size_min, float step_size_max, bool usemultthread
    );

    RRTResult SE3runRRT();

private:

    // Auxiliary methods

    VectorXf SE3sampleRandomConfig(); 

    VectorXf SE3steer(const VectorXf& q_near, const VectorXf& q_rand); 

    bool SE3isPathCollisionFree(const VectorXf& q1, const VectorXf& q2); 

    bool SE3reachedGoal(const VectorXf& q_new);

    void SE3addNode(const VectorXf& q_new, int parent_idx); 

    int SE3getNearestNeighbor(const VectorXf& q_rand); 

    vector<VectorXf> SE3backtrackPath(const int &goal_idx); 

    vector<VectorXf> SE3shortcutting(const vector<VectorXf>& path);
    
    vector<VectorXf> SE3interpolatePath(const vector<VectorXf>& path);

    bool SE3verifyPath(const vector<VectorXf>& path);

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