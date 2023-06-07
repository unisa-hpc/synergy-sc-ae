#ifdef __ENABLED_SYNERGY
    #include <synergy.hpp>
    #define selected_queue synergy::queue
#else
    #define selected_queue sycl::queue
#endif

