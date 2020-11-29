cmake_minimum_required (VERSION 3.17.0)
project (ParallelPanorama)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Get GIT information
find_package(Git REQUIRED)
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY "${local_dir}"
        OUTPUT_VARIABLE GIT_COMMIT_HASH
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    message( STATUS "Git commit: ${GIT_COMMIT_HASH}")
else()
    set(GIT_VERSION_STRING "Git not found")
endif()

# Open source libraries
find_package(OpenCV REQUIRED)

add_subdirectory(src)