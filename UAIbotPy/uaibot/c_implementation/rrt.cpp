#include "rrt.hpp"
#include <iostream>
#include <random>


using namespace std;
using namespace Eigen;

float randomFloat(float a, float b) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    std::uniform_real_distribution<float> dist(a, b);
    
    return dist(gen);
}

RRT::RRT(Manipulator robot, VectorXf q_start, vector<VectorXf> q_goal, Matrix4f htm, vector<GeometricPrimitives> obstacles, 
    int max_iter, float goal_tolerance, float goal_bias, float step_size_min, float step_size_max, bool usemultthread) {   
    
    this->robot = robot;
    this->q_start = q_start;
    this->q_goal = q_goal; 
    this->obstacles = obstacles;
    this->max_iter = max_iter;
    this->goal_tolerance = goal_tolerance;
    this->goal_bias = goal_bias;

    this->step_size_min = step_size_min;
    this->step_size_max = step_size_max;

    this->usemultthread = usemultthread;

    this->result.success = false;
    this->result.iterations = 0;
    this->result.planning_time = 0;
    this->result.path = vector<VectorXf>();

    this->n = robot.no_links;
    
    this->q_min = robot.q_min;
    this->q_max = robot.q_max;

    this->htm = htm;


    //////////////////////////
    // KDtree
    //////////////////////////

    this->pointcloud = std::make_unique<RRTPointCloud>();

    this->kdtree = std::make_unique<KDTreeAdaptor>( this->n, *(this->pointcloud), nanoflann::KDTreeSingleIndexAdaptorParams(10));

    this->addNode(this->q_start , -1 );
}

VectorXf RRT::sampleRandomConfig() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    float r = dist01(gen);
    if (r < this->goal_bias) {
        std::uniform_int_distribution<int> goal_dist(0, this->q_goal.size() - 1);
        return this->q_goal[goal_dist(gen)];
    }

    VectorXf q_rand(n);
    for (int i = 0; i < n; ++i) {
        float min_val = q_min(i);
        float max_val = q_max(i);
        std::uniform_real_distribution<float> dist(min_val, max_val);
        q_rand(i) = dist(gen);
    }
    return q_rand;
}

RRTResult RRT::runRRT() {
    auto start_time = std::chrono::high_resolution_clock::now();
    vector<VectorXf> path;
    // reset result and found flag
    this->result.success = false;
    this->found.store(false);
    if (this->usemultthread) {
        std::atomic<int> iter_count(0);
        int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        // worker lambda
        auto worker = [&]() {
            while (true) {
                int i = iter_count.fetch_add(1);
                if (i > this->max_iter || this->found.load()) break;
                // sample and steer
                VectorXf q_rand = this->sampleRandomConfig();
                int nearest_idx;
                VectorXf q_near;
                { // protect tree and kdtree access
                    std::lock_guard<std::mutex> lock(this->tree_mutex);
                    nearest_idx = this->getNearestNeighbor(q_rand);
                    q_near = this->tree[nearest_idx].q;
                }
                VectorXf q_new = this->steer(q_near, q_rand);
                if (this->isPathCollisionFree(q_near, q_new)) {
                    std::lock_guard<std::mutex> lock(this->tree_mutex);
                    this->addNode(q_new, nearest_idx);
                    if (!this->found.load() && this->reachedGoal(q_new)) {
                        // find closest goal
                        auto closest_goal = std::min_element(
                            this->q_goal.begin(), this->q_goal.end(),
                            [&q_new](const VectorXf& a, const VectorXf& b) { return (q_new - a).norm() < (q_new - b).norm(); }
                        );
                        if (this->isPathCollisionFree(q_new, *closest_goal)) {
                            this->addNode(*closest_goal, static_cast<int>(this->tree.size() - 1));
                            this->found.store(true);
                            path = this->backtrackPath(static_cast<int>(this->tree.size() - 1));
                        }
                    }
                }
            }
        };
        // launch threads
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(worker);
        }
        for (auto& th : threads) th.join();
        // set results
        this->result.iterations = iter_count.load();
        this->result.success = this->found.load();
        if (this->result.success) {
            this->result.path = this->shortcutting(path);
            this->result.path = this->interpolatePath(this->result.path);
        }
    } else {
        // single-threaded execution
        for (size_t i = 0; i <= this->max_iter; i++) {
            this->result.iterations = static_cast<int>(i);
            VectorXf q_rand = this->sampleRandomConfig();
            int nearest_idx = this->getNearestNeighbor(q_rand);
            VectorXf q_near = this->tree[nearest_idx].q;
            VectorXf q_new = this->steer(q_near, q_rand);
            if (this->isPathCollisionFree(q_near, q_new)) {
                this->addNode(q_new, nearest_idx);
                if (this->reachedGoal(q_new)) {
                    auto closest_goal = std::min_element(
                        this->q_goal.begin(), this->q_goal.end(),
                        [&q_new](const VectorXf& a, const VectorXf& b) { return (q_new - a).norm() < (q_new - b).norm(); }
                    );
                    if (this->isPathCollisionFree(q_new, *closest_goal)) {
                        this->addNode(*closest_goal, static_cast<int>(this->tree.size() - 1));
                        this->result.success = true;
                        path = this->backtrackPath(static_cast<int>(this->tree.size() - 1));
                        break;
                    }
                }
            }
        }
        if (this->result.success) {
            this->result.path = this->shortcutting(path);
            this->result.path = this->interpolatePath(this->result.path);
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> planning_duration = end_time - start_time;
    this->result.planning_time = planning_duration.count();

    if( (this->result.success) && (this->verifyPath(this->result.path) == false) ) {
        this->result.success = false;
        this->result.path.clear();
    }

    return this->result;
}

VectorXf RRT::steer(const VectorXf &q_near, const VectorXf &q_rand)
{
    VectorXf diff = q_rand - q_near;
    float distance = diff.norm();


    float size = randomFloat(this->step_size_min , this->step_size_max);

    if (distance <= size) {
        return q_rand;
    }

    VectorXf direction = diff / distance;
    VectorXf q_new = q_near + size * direction;
    return q_new;
}

bool RRT::isPathCollisionFree(const VectorXf &q1, const VectorXf &q2)
{

    float alpha = 0;
    VectorXf q_interp;
    CheckFreeConfigResult check;
    int temp = ceil((q2 - q1).norm() / SAMPLE_STEP);
    int no_samples = temp > 10 ? temp : 10;

    

    for (   int i = 0; i <= no_samples; i++) {
        alpha = ((float) i) / ((float) no_samples);
        q_interp = (1 - alpha) * q1 + alpha * q2;
        check = robot.check_free_configuration(q_interp, this->htm, obstacles, true, true, 0.0005, 0.005, 20);
        if (check.isfree == false){
            return false;
        }
    }
    return true;
}

bool RRT::reachedGoal(const VectorXf &q_new) {
    for (const auto& goal : this->q_goal) {
        if ((q_new - goal).norm() <= this->goal_tolerance) {
            return true;
        }
    }
    return false;
}

void RRT::addNode(const VectorXf &q_new, int parent_idx) {
    this->tree.push_back(Node(q_new, parent_idx));

    RRTPointCloud::Point p;
    p.coords.resize(this->n);
    for (int i = 0; i < this->n; i++) {
        p.coords[i] = q_new[i];
    }

    this->pointcloud->pts.push_back(std::move(p));
    this->kdtree->buildIndex();
}

int RRT::getNearestNeighbor(const VectorXf &q_rand){
    
    std::vector<float> query_pt(this->n);
    for (int i = 0; i < this->n; i++) {query_pt[i] = q_rand[i];}
    size_t ret_index;
    float out_dist_sqr;

    nanoflann::KNNResultSet<float> resultSet(1); 
    resultSet.init(&ret_index, &out_dist_sqr);

    this->kdtree->findNeighbors(resultSet, query_pt.data(), nanoflann::SearchParameters(10));

    return static_cast<int>(ret_index);
}

vector<VectorXf> RRT::backtrackPath(const int &goal_idx){
    vector<VectorXf> path;
    int curr_idx = goal_idx;
    VectorXf curr_q;

    while (curr_idx != -1){
        curr_q = this->tree[curr_idx].q;
        path.push_back(curr_q);
        curr_idx = this->tree[curr_idx].parent_idx;
    }
    
    reverse(path.begin(), path.end());
    return path;
}



vector<VectorXf> RRT::shortcutting(const vector<VectorXf>& path){

    if (path.size() <= 2) return path;
    vector<VectorXf> best_path = path;
    bool improved = true;
    while (improved) {
        improved = false;
        for (size_t i = 0; i < best_path.size() - 2; ++i) {
            for (size_t j = i + 2; j < best_path.size(); ++j) {
                if (isPathCollisionFree(best_path[i], best_path[j])) {
                    vector<VectorXf> new_path;
                    new_path.insert(new_path.end(), best_path.begin(), best_path.begin() + i + 1);
                    new_path.insert(new_path.end(), best_path.begin() + j, best_path.end());
                    best_path = new_path;
                    improved = true;
                    break; 
                }
            }
            if (improved) break;
        }
    }
    return best_path;
}

bool RRT::verifyPath(const vector<VectorXf> &path){
    for ( auto q : path ){
        CheckFreeConfigResult check = robot.check_free_configuration(q, this->htm, obstacles, true, true, 0.0005, 0.005, 20);
        if (check.isfree == false){                 
            return false; 
        }
    }
    return true;
}

vector<VectorXf> RRT::interpolatePath(const vector<VectorXf> &path){
    vector<VectorXf> new_path;

    for (size_t i = 0; i < path.size() - 1; i++){
        VectorXf q_start = path[i];
        VectorXf q_end = path[i+1];
        int num_steps = static_cast<int>((q_end - q_start).norm() / INTERP_STEP);

        for (size_t j = 1; j <= num_steps; j++){
            float alpha = static_cast<float>(j) / static_cast<float>(num_steps);
            VectorXf q_interp = (1 - alpha) * q_start + alpha * q_end;
            new_path.push_back(q_interp);
        }
    }


    return new_path;
}