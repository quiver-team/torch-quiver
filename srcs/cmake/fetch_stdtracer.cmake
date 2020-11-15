SET(STDTRACER_GIT_URL
    https://github.com/stdml/stdtracer.git
    CACHE STRING "URL for clone stdtracer")

SET(STDTRACER_GIT_TAG
    "master"
    CACHE STRING "git tag for checkout stdtracer")

INCLUDE(FetchContent)

FETCHCONTENT_DECLARE(
    stdtracer
    GIT_REPOSITORY ${STDTRACER_GIT_URL}
    GIT_TAG ${STDTRACER_GIT_TAG})

FETCHCONTENT_POPULATE(stdtracer)
INCLUDE_DIRECTORIES(_deps/stdtracer-src/include)
