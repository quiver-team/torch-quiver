FUNCTION(BUILD_QUIVER_LIB target)
    ADD_LIBRARY(${target} SHARED)
    TARGET_SOURCE_TREE(${target} ${CMAKE_SOURCE_DIR}/srcs/cpp/src/*.cpp)

    IF(ENABLE_CUDA)
        TARGET_SOURCE_TREE(${target} ${CMAKE_SOURCE_DIR}/srcs/cpp/src/*.cu)
        TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${CUDA_INCLUDE_DIRS})
        TARGET_LINK_LIBRARIES(${target} ${CUDA_LIBRARIES})
        TARGET_COMPILE_OPTIONS(
            ${target}
            PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
    ENDIF()
ENDFUNCTION()

BUILD_QUIVER_LIB(quiver)
