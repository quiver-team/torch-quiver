ENABLE_TESTING()

FIND_PACKAGE(GTest REQUIRED)
FIND_PACKAGE(Threads REQUIRED)

FUNCTION(LINK_GTEST target)
    TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${GTEST_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(${target} ${GTEST_BOTH_LIBRARIES} Threads::Threads)
ENDFUNCTION()

FUNCTION(ADD_UNIT_TEST target)
    ADD_EXECUTABLE(${target} ${ARGN} ${CMAKE_SOURCE_DIR}/tests/cpp/main.cpp)
    LINK_GTEST(${target})
    ADD_TEST(NAME ${target} COMMAND ${target})
    TARGET_COMPILE_DEFINITIONS(${target} PRIVATE -DQUIVER_TEST=1)
    TARGET_COMPILE_OPTIONS(
        ${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
ENDFUNCTION()

FILE(GLOB tests ${CMAKE_SOURCE_DIR}/tests/cpp/test_*.cpp)

FOREACH(t ${tests})
    GET_FILENAME_COMPONENT(name ${t} NAME_WE)
    STRING(REPLACE "_" "-" name ${name})
    ADD_UNIT_TEST(${name} ${t})
    TARGET_INCLUDE_DIRECTORIES(${name} PRIVATE ${CUDA_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(${name} ${CUDA_LIBRARIES})
ENDFOREACH()

FILE(GLOB tests ${CMAKE_SOURCE_DIR}/tests/cpp/test_*.cu)

FOREACH(t ${tests})
    GET_FILENAME_COMPONENT(name ${t} NAME_WE)
    STRING(REPLACE "_" "-" name ${name})
    ADD_UNIT_TEST(${name} ${t})
ENDFOREACH()
