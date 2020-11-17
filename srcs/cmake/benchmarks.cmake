ADD_LIBRARY(bench-common benchmarks/cpp/common.cpp)

FUNCTION(ADD_BENCH_BIN target)
    GET_FILENAME_COMPONENT(name ${target} NAME_WE)
    STRING(REPLACE "_" "-" name ${name})
    ADD_EXECUTABLE(${name} ${target})
    TARGET_LINK_LIBRARIES(${name} bench-common)
    TARGET_COMPILE_OPTIONS(
        ${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
ENDFUNCTION()

ADD_BENCH_BIN(benchmarks/cpp/bench_quiver_cpu.cpp)
ADD_BENCH_BIN(benchmarks/cpp/bench_quiver_gpu.cu)
ADD_BENCH_BIN(benchmarks/cpp/bench_async_algo.cu)
