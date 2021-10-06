INCLUDE(srcs/cmake/generated/torch.cmake)

FUNCTION(build_torch_quiver_ext target)
    ADD_LIBRARY(${target} SHARED)
    # ADD_LIBRARY(${target} STATIC)
    TARGET_INCLUDE_DIRECTORIES(${target}
                               PRIVATE ${CMAKE_SOURCE_DIR}/srcs/cpp/include)
    TARGET_SOURCE_DIR(${target} ${CMAKE_SOURCE_DIR}/srcs/cpp/src/quiver/*.cpp)
    TARGET_SOURCE_DIR(${target}
                      ${CMAKE_SOURCE_DIR}/srcs/cpp/src/quiver/cpu/*.cpp)
    IF(ENABLE_CUDA)
        TARGET_SOURCE_DIR(${target}
                          ${CMAKE_SOURCE_DIR}/srcs/cpp/src/quiver/cuda/*.cpp)
        TARGET_SOURCE_DIR(${target}
                          ${CMAKE_SOURCE_DIR}/srcs/cpp/src/quiver/cuda/*.cu)
        TARGET_SET_CUDA_OPTIONS(${target})
    ENDIF()
ENDFUNCTION()

BUILD_TORCH_QUIVER_EXT(torch_quiver)
