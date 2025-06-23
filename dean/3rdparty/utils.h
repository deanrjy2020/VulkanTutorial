#ifndef UTILS_H
#define UTILS_H

//====================================================================
// PRINTF

#define ENABLE_PRINT 1
#if ENABLE_PRINT
#define PRINTF(...)                            \
    do {                                       \
        printf(__VA_ARGS__);                   \
        fflush(stdout);                        \
    } while (0)
#else
#define PRINTF(...)
#endif

//====================================================================
// dump image
// 见: 31.2_compute_shader_rt_bvh.cpp::dumpSwapchainImage()

//====================================================================
// Others

#endif  // UTILS_H
