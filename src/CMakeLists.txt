file(GLOB SRC_LIST "*.h" "*.c" "*.hpp" "*.cpp")
message(STATUS "Sources: ${SRC_LIST}")

include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

add_executable(ParallelPanorama ${SRC_LIST})

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(TARGET ParallelPanorama PROPERTY CXX_STANDARD 17)


target_link_libraries(ParallelPanorama ${OpenCV_LIBS} )

set(OpenCV_RUNTIME_LIBS ${OpenCV_INSTALL_PATH}/x64/vc15/bin/opencv_videoio_ffmpeg420_64.dll;
                        ${OpenCV_INSTALL_PATH}/x64/vc15/bin/opencv_world420.dll)

# Install binaries and files
install(TARGETS ParallelPanorama RUNTIME DESTINATION . COMPONENT applications)
install(FILES ${OpenCV_RUNTIME_LIBS}
        DESTINATION .
        COMPONENT dependencies)
if(EXE_DEST_DIR)
    string(REPLACE "\\"
                   "/"
                   DEST_DIR
                   "${EXE_DEST_DIR}")

    add_custom_command(TARGET ParallelPanorama POST_BUILD
                       COMMAND ${CMAKE_COMMAND}
                               -E
                               copy_if_different
                               $<TARGET_FILE:ParallelPanorama>
                               "${DEST_DIR}/ParallelPanorama.exe")
endif()

# CPack Setup
# CPACK macros below here
set(CPACK_PACKAGE_NAME "ParallelPanorama")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Group project for ECE5510")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "ParallelPanorama")

# Define components and their display names
set(CPACK_COMPONENTS_ALL
    applications
    dependencies
    config
    cmds)
set(CPACK_COMPONENT_APPLICATIONS_DISPLAY_NAME "ParallelPanorama")

# Define groups
set(CPACK_COMPONENT_APPLICATIONS_GROUP "application")
set(CPACK_COMPONENT_DEPENDENCIES_GROUP "libs")

# Define NSIS installation types
set(CPACK_ALL_INSTALL_TYPES Full)
set(CPACK_COMPONENT_APPLICATIONS_INSTALL_TYPES Full)
set(CPACK_COMPONENT_DEPENDENCIES_INSTALL_TYPES Full)
set(CPACK_COMPONENT_CONFIG_INSTALL_TYPES Full)

# Must be after the last CPACK macros
set(CPACK_OUTPUT_FILE_PREFIX "")
set(CPACK_GENERATOR "ZIP" CACHE STRING "Generators to support. semi-colon delimited list")
include(CPack)