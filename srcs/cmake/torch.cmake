INCLUDE(srcs/cmake/generated/torch.cmake)

# ADD_LIBRARY(torch_quiver SHARED)
ADD_LIBRARY(torch_quiver STATIC)
TARGET_INCLUDE_DIRECTORIES(torch_quiver
                           PUBLIC ${CMAKE_SOURCE_DIR}/srcs/cpp/include)

TARGET_SOURCE_DIR(torch_quiver ${CMAKE_SOURCE_DIR}/srcs/cpp/src/quiver/*.cpp)
TARGET_SOURCE_DIR(torch_quiver
                  ${CMAKE_SOURCE_DIR}/srcs/cpp/src/quiver/cpu/*.cpp)

IF(ENABLE_CUDA)
    ENABLE_LANGUAGE(CUDA)
    INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/include)
    LINK_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    TARGET_COMPILE_OPTIONS(
        torch_quiver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
    TARGET_SOURCE_DIR(torch_quiver
                      ${CMAKE_SOURCE_DIR}/srcs/cpp/src/quiver/cuda/*.cpp)
    TARGET_SOURCE_DIR(torch_quiver
                      ${CMAKE_SOURCE_DIR}/srcs/cpp/src/quiver/cuda/*.cu)
ENDIF()
