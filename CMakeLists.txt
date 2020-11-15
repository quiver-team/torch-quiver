CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(pytorch-quiver)

CMAKE_POLICY(SET CMP0104 OLD) # FIXME: set CUDA_ARCHITECTURES

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)
SET(LIBRARY_OUTPUT_PATH lib)
SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/srcs/cmake)

OPTION(USE_STRICT "Enable strict build options." OFF)
IF(USE_STRICT)
    ADD_DEFINITIONS(-Wall)
    ADD_DEFINITIONS(-Werror)
    ADD_DEFINITIONS(-Wfatal-errors)
ENDIF()

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/srcs/cpp/include)

IF(ENABLE_TRACE)
    INCLUDE(srcs/cmake/fetch_stdtracer.cmake)
ENDIF()
IF(ENABLE_CUDA)
    INCLUDE(srcs/cmake/cuda.cmake)
ENDIF()
IF(BUILD_TORCH_EXTENSION)
    INCLUDE(srcs/cmake/torch.cmake)
ENDIF()

FUNCTION(ADD_SOURCE target)
    TARGET_SOURCES(${target} PRIVATE ${ARGN})
ENDFUNCTION()

# ADD_LIBRARY(quiver SHARED) ADD_SOURCE(quiver
# ${CMAKE_SOURCE_DIR}/srcs/cpp/src/quiver/quiver.cpp) ADD_SOURCE(quiver
# ${CMAKE_SOURCE_DIR}/srcs/cpp/src/quiver/trace.cpp) IF(ENABLE_CUDA)
# ADD_SOURCE(quiver ${CMAKE_SOURCE_DIR}/srcs/cpp/src/quiver/cuda/quiver.cu)
# ENDIF()

IF(ENABLE_CUDA)
    # ADD_EXECUTABLE(cuda-sample examples/cuda_sample.cpp)
ENDIF()

IF(BUILD_TESTS)
    INCLUDE(srcs/cmake/tests.cmake)
ENDIF()

IF(BUILD_BENCHMARKS)
    INCLUDE(srcs/cmake/benchmarks.cmake)
ENDIF()
