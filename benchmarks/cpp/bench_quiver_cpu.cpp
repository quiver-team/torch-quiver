#include <cstdio>
#include <iostream>
#include <map>
#include <vector>

#include "common.hpp"
#include <quiver/trace.hpp>

DEFINE_TRACE_CONTEXTS;

int main()
{
    TRACE_SCOPE(__func__);
    using W = float;
    auto g = gen_random_graph(100, 1000);
    // TODO
    return 0;
}
