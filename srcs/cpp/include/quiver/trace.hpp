#pragma once
#include <chrono>
#include <cstdio>
#include <string>

#ifdef QUIVER_ENABLE_TRACE
    #include <tracer/simple>
#else
    #define TRACE_SCOPE(name)
    #define LOG_SCOPE(name)
    #define TRACE_STMT(e) e;
    #define TRACE_EXPR(e) e
    #define DEFINE_TRACE_CONTEXTS
#endif
