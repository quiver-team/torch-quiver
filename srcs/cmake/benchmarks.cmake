ADD_LIBRARY(bench-common benchmarks/cpp/common.cpp)

ADD_EXECUTABLE(bench-quiver-cpu benchmarks/cpp/bench_quiver_cpu.cpp)
ADD_EXECUTABLE(bench-quiver-gpu benchmarks/cpp/bench_quiver_gpu.cu)
TARGET_LINK_LIBRARIES(bench-quiver-cpu bench-common)
TARGET_LINK_LIBRARIES(bench-quiver-gpu bench-common)
