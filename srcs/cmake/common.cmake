FUNCTION(TARGET_SOURCE_TREE target)
    FILE(GLOB_RECURSE SRCS ${ARGN})
    TARGET_SOURCES(${target} PRIVATE ${SRCS})
ENDFUNCTION()
