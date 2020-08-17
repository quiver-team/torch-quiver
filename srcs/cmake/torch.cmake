INCLUDE(srcs/cmake/generated/torch.cmake)

FUNCTION(TARGET_SOURCES_GLOB target)
    FILE(GLOB SRCS ${ARGN})
    TARGET_SOURCES(${target} PRIVATE ${SRCS})
ENDFUNCTION()

# ADD_LIBRARY(torch_quiver SHARED)
ADD_LIBRARY(torch_quiver STATIC)
TARGET_INCLUDE_DIRECTORIES(torch_quiver
                           PUBLIC ${CMAKE_SOURCE_DIR}/srcs/cpp/include)

TARGET_SOURCES_GLOB(torch_quiver ${CMAKE_SOURCE_DIR}/srcs/cpp/src/quiver/*.cpp)
TARGET_SOURCES_GLOB(torch_quiver
                    ${CMAKE_SOURCE_DIR}/srcs/cpp/src/quiver/cpu/*.cpp)

IF(ENABLE_CUDA)
    ENABLE_LANGUAGE(CUDA)
    INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/include)
    LINK_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

    TARGET_SOURCES_GLOB(torch_quiver
                        ${CMAKE_SOURCE_DIR}/srcs/cpp/src/quiver/cuda/*.cpp)
    TARGET_SOURCES_GLOB(torch_quiver
                        ${CMAKE_SOURCE_DIR}/srcs/cpp/src/quiver/cuda/*.cu)
ENDIF()
