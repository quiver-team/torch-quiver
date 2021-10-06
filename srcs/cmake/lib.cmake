FUNCTION(BUILD_QUIVER_LIB target)
    ADD_LIBRARY(${target} SHARED)
    TARGET_SOURCE_TREE(${target} ${CMAKE_SOURCE_DIR}/srcs/cpp/src/*.cpp)

    IF(ENABLE_CUDA)
        TARGET_SOURCE_TREE(${target} ${CMAKE_SOURCE_DIR}/srcs/cpp/src/*.cu)
        TARGET_SET_CUDA_OPTIONS(${target})
    ENDIF()
ENDFUNCTION()

BUILD_QUIVER_LIB(quiver)
