// tensor_avx_init.c

#include "tensor_avx_init.h"
#include <stdio.h>      // fprintf
#include <cpuid.h>      // __get_cpuid_count, bit_AVX2
#include <omp.h>        // OpenMP: omp_get_num_procs, omp_set_num_threads
#include <stdbool.h>    // bool

// Check for AVX2 support at runtime
int has_avx2() {
    unsigned int eax, ebx, ecx, edx;

    // Call CPUID with eax = 0 to get highest valid function ID
    if (!__get_cpuid_max(0, NULL)) return 0;

    // Call CPUID with eax=7, ecx=0 to get extended features
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
        return 0;

    return (ebx & bit_AVX2) != 0;
}

// Initialize tensor operations library
int tensor_ops_init() {
    if (!has_avx2()) {
        fprintf(stderr, "Error: AVX2 not supported on this CPU\n");
        return -1;
    }
    
    // Use fewer threads than available cores to avoid oversubscription
    // This prevents thread contention and context switching overhead
    int num_procs = omp_get_num_procs();
    int num_threads = num_procs > 4 ? num_procs - 2 : num_procs;
    
    // For small matrices, too many threads create more overhead than benefit
    // Limit to 4-8 threads for typical deep learning workloads
    if (num_threads > 8) num_threads = 8;
    
    omp_set_num_threads(num_threads);
    
    // Use dynamic scheduling with small chunk size for better load balancing
    #ifdef _OPENMP
    omp_set_schedule(omp_sched_dynamic, 16);
    #endif
    
    fprintf(stderr, "SIMD Tensor Backend initialized with %d threads\n", num_threads);
    
    return 0;
}