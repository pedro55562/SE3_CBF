#include "SE3rrt.hpp"
#include <iostream>
#include <random>

using namespace std;
using namespace Eigen;


float SE3randomFloat(float a, float b) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    std::uniform_real_distribution<float> dist(a, b);
    
    return dist(gen);
}

float SE3_distance(const VectorXf &q1, const VectorXf &q2) {
    VectorXf s1(3), s2(3);
    s1 << q1(0), q1(1), q1(2);
    s2 << q2(0), q2(1), q2(2);

    float s_dist = (s2 - s1).norm();

    VectorXf gamma1(3), gamma2(3);

    gamma1 << q1(3), q1(4), q1(5);
    gamma2 << q2(3), q2(4), q2(5);

    float gamma_dist = (1 - cos(gamma2(0) - gamma1(0))) + (1 - cos(gamma2(1) - gamma1(1))) + (1 - cos(gamma2(2) - gamma1(2)));
    return s_dist + gamma_dist;
}


inline float wrapToPi(float angle)
{
    while (angle > M_PI)  angle -= 2.0f * M_PI;
    while (angle < -M_PI) angle += 2.0f * M_PI;
    return angle;
}

inline float interpAngle(float a1, float a2, float alpha)
{
    float delta = wrapToPi(a2 - a1);
    return wrapToPi(a1 + alpha * delta);
}

std::vector<VectorXf> interpolateSE3(
    const VectorXf &q1,
    const VectorXf &q2,
    float step)
{
    std::vector<VectorXf> path;

    VectorXf s1(3), s2(3);
    VectorXf gamma1(3), gamma2(3);

    s1 << q1(0), q1(2), q1(4);
    s2 << q2(0), q2(2), q2(4);

    gamma1 << q1(1), q1(3), q1(5);
    gamma2 << q2(1), q2(3), q2(5);

    int no_samples = std::max(
        2,
        static_cast<int>(std::ceil(SE3_distance(q1, q2) / step))
    );

    path.reserve(no_samples + 1);

    for (int i = 0; i <= no_samples; ++i)
    {
        float alpha = static_cast<float>(i) / no_samples;

        // --- Interpolação linear ---
        VectorXf s = (1.0f - alpha) * s1 + alpha * s2;

        // --- Interpolação angular ---
        VectorXf gamma(3);
        for (int k = 0; k < 3; ++k)
            gamma(k) = interpAngle(gamma1(k), gamma2(k), alpha);

        // --- Monta configuração ---
        VectorXf q_interp(6);
        q_interp <<
            s(0), gamma(0),
            s(1), gamma(1),
            s(2), gamma(2);

        path.push_back(q_interp);
    }

    return path;
}




SE3RRT::SE3RRT(Manipulator robot, VectorXf q_start, vector<VectorXf> q_goal, Matrix4f htm, vector<GeometricPrimitives> obstacles, 
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

    this->SE3addNode(this->q_start , -1 );
}

VectorXf SE3RRT::SE3sampleRandomConfig() {
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

RRTResult SE3RRT::SE3runRRT() {
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
                // sample and SE3steer
                VectorXf q_rand = this->SE3sampleRandomConfig();
                int nearest_idx;
                VectorXf q_near;
                { // protect tree and kdtree access
                    std::lock_guard<std::mutex> lock(this->tree_mutex);
                    nearest_idx = this->SE3getNearestNeighbor(q_rand);
                    q_near = this->tree[nearest_idx].q;
                }
                VectorXf q_new = this->SE3steer(q_near, q_rand);
                if (this->SE3isPathCollisionFree(q_near, q_new)) {
                    std::lock_guard<std::mutex> lock(this->tree_mutex);
                    this->SE3addNode(q_new, nearest_idx);
                    if (!this->found.load() && this->SE3reachedGoal(q_new)) {
                        // find closest goal
                        auto closest_goal = std::min_element(
                            this->q_goal.begin(), this->q_goal.end(),
                            [&q_new](const VectorXf& a, const VectorXf& b) { return SE3_distance(q_new, a) < SE3_distance(q_new, b); }
                        );                                                        
                        if (this->SE3isPathCollisionFree(q_new, *closest_goal)) {
                            this->SE3addNode(*closest_goal, static_cast<int>(this->tree.size() - 1));
                            this->found.store(true);
                            path = this->SE3backtrackPath(static_cast<int>(this->tree.size() - 1));
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
            this->result.path = this->SE3shortcutting(path);
            this->result.path = this->SE3interpolatePath(this->result.path);
        }
    } else {
        // single-threaded execution
        for (size_t i = 0; i <= this->max_iter; i++) {
            this->result.iterations = static_cast<int>(i);
            VectorXf q_rand = this->SE3sampleRandomConfig();
            int nearest_idx = this->SE3getNearestNeighbor(q_rand);
            VectorXf q_near = this->tree[nearest_idx].q;
            VectorXf q_new = this->SE3steer(q_near, q_rand);
            if (this->SE3isPathCollisionFree(q_near, q_new)) {
                this->SE3addNode(q_new, nearest_idx);
                if (this->SE3reachedGoal(q_new)) {
                    auto closest_goal = std::min_element(
                        this->q_goal.begin(), this->q_goal.end(),
                        [&q_new](const VectorXf& a, const VectorXf& b) { return SE3_distance(q_new, a) < SE3_distance(q_new, b); }
                    );
                    if (this->SE3isPathCollisionFree(q_new, *closest_goal)) {
                        this->SE3addNode(*closest_goal, static_cast<int>(this->tree.size() - 1));
                        this->result.success = true;
                        path = this->SE3backtrackPath(static_cast<int>(this->tree.size() - 1));
                        break;
                    }
                }
            }
        }
        if (this->result.success) {
            this->result.path = this->SE3shortcutting(path);
            this->result.path = this->SE3interpolatePath(this->result.path);
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> planning_duration = end_time - start_time;
    this->result.planning_time = planning_duration.count();

    if( (this->result.success) && (this->SE3verifyPath(this->result.path) == false) ) {
        this->result.success = false;
        this->result.path.clear();
    }

    return this->result;
}

VectorXf SE3RRT::SE3steer(const VectorXf &q_near, const VectorXf &q_rand)
{
    VectorXf diff = q_rand - q_near;
    float distance = SE3_distance(q_rand, q_near);


    float size = SE3randomFloat(this->step_size_min , this->step_size_max);

    if (distance <= size) {
        return q_rand;
    }

    VectorXf direction = diff / distance;
    VectorXf q_new = q_near + size * direction;
    return q_new;
}

bool SE3RRT::SE3isPathCollisionFree( const VectorXf &q1, const VectorXf &q2) {
    vector<VectorXf> configs = interpolateSE3(q1, q2, SAMPLE_STEP);

    for (const VectorXf &q : configs)
    {
        CheckFreeConfigResult check =
            robot.check_free_configuration(
                q, htm, obstacles,
                true, true, 0.0005, 0.005, 20
            );

        if (check.isfree == false)
            return false;
    }

    return true;
}



bool SE3RRT::SE3reachedGoal(const VectorXf &q_new) {
    for (const auto& goal : this->q_goal) {
        if (SE3_distance(q_new, goal) <= this->goal_tolerance) {
            return true;
        }
    }
    return false;
}

void SE3RRT::SE3addNode(const VectorXf &q_new, int parent_idx) {
    this->tree.push_back(Node(q_new, parent_idx));

    RRTPointCloud::Point p;
    p.coords.resize(this->n);
    for (int i = 0; i < this->n; i++) {
        p.coords[i] = q_new[i];
    }

    this->pointcloud->pts.push_back(std::move(p));
    this->kdtree->buildIndex();
}

int SE3RRT::SE3getNearestNeighbor(const VectorXf &q_rand){
    
    std::vector<float> query_pt(this->n);
    for (int i = 0; i < this->n; i++) {query_pt[i] = q_rand[i];}
    size_t ret_index;
    float out_dist_sqr;

    nanoflann::KNNResultSet<float> resultSet(1); 
    resultSet.init(&ret_index, &out_dist_sqr);

    this->kdtree->findNeighbors(resultSet, query_pt.data(), nanoflann::SearchParameters(10));

    return static_cast<int>(ret_index);
}

vector<VectorXf> SE3RRT::SE3backtrackPath(const int &goal_idx){
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



vector<VectorXf> SE3RRT::SE3shortcutting(const vector<VectorXf>& path){

    if (path.size() <= 2) return path;
    vector<VectorXf> best_path = path;
    bool improved = true;
    while (improved) {
        improved = false;
        for (size_t i = 0; i < best_path.size() - 2; ++i) {
            for (size_t j = i + 2; j < best_path.size(); ++j) {
                if (SE3isPathCollisionFree(best_path[i], best_path[j])) {
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

bool SE3RRT::SE3verifyPath(const vector<VectorXf> &path){
    for ( auto q : path ){
        CheckFreeConfigResult check = robot.check_free_configuration(q, this->htm, obstacles, true, true, 0.0005, 0.005, 20);
        if (check.isfree == false){                 
            return false; 
        }
    }
    return true;
}

std::vector<Eigen::VectorXf>
SE3RRT::SE3interpolatePath(
    const std::vector<Eigen::VectorXf> &path)
{
    std::vector<Eigen::VectorXf> new_path;

    if (path.size() < 2)
        return new_path;

    for (size_t i = 0; i < path.size() - 1; ++i)
    {
        const Eigen::VectorXf &q_start = path[i];
        const Eigen::VectorXf &q_end   = path[i + 1];

        // Interpola o segmento inteiro usando o método SE(3)
        std::vector<Eigen::VectorXf> segment =
            interpolateSE3(q_start, q_end, INTERP_STEP);

        /*
         * Evita duplicar o ponto inicial:
         * - para o primeiro segmento, copia tudo
         * - para os demais, ignora o primeiro ponto
         */
        size_t start_index = (i == 0) ? 0 : 1;

        for (size_t j = start_index; j < segment.size(); ++j)
        {
            new_path.push_back(segment[j]);
        }
    }

    return new_path;
}
