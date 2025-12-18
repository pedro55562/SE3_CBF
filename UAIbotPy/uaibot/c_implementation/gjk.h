// #ifndef GJK_H
// #define GJK_H

// #define GJK_MAX_ITERATIONS 20

// struct gjk_support {
//     int aid, bid;   /* in */
//     float a[3];     /* in */
//     float b[3];     /* in */
//     float da[3];    /* out */
//     float db[3];    /* out */
// };
// struct gjk_simplex {
//     int max_iter, iter;
//     int hit, cnt;
//     struct gjk_vertex {
//         float a[3];
//         float b[3];
//         float p[3];
//         int aid, bid;
//     } v[4];
//     float bc[4], D;
// };
// struct gjk_result {
//     int hit;
//     float p0[3];
//     float p1[3];
//     float distance_squared;
//     int iterations;
// };
// extern int gjk(struct gjk_simplex *s, struct gjk_support *sup);
// extern void gjk_analyze(struct gjk_result *res, const struct gjk_simplex *s);
// extern void gjk_quad(struct gjk_result *res, float a_radius, float b_radius);

// #endif

#ifndef GJK_H
#define GJK_H

#ifdef __cplusplus
extern "C" {
#endif

#define GJK_MAX_ITERATIONS 20

struct gjk_support {
    int aid, bid;   /* in */
    float a[3];     /* in */
    float b[3];     /* in */
    float da[3];    /* out */
    float db[3];    /* out */
};

struct gjk_simplex {
    int max_iter, iter;
    int hit, cnt;
    struct gjk_vertex {
        float a[3];
        float b[3];
        float p[3];
        int aid, bid;
    } v[4];
    float bc[4], D;
    int error;
};

struct gjk_result {
    int hit;
    float p0[3];
    float p1[3];
    float distance_squared;
    int iterations;
};

/* Ensure these functions are not name-mangled in C++ */
extern int gjk(struct gjk_simplex *s, struct gjk_support *sup);
extern void gjk_analyze(struct gjk_result *res, const struct gjk_simplex *s);
extern void gjk_quad(struct gjk_result *res, float a_radius, float b_radius);

#ifdef __cplusplus
}
#endif

#endif /* GJK_H */
