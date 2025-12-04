cmake_minimum_required(VERSION 3.15)
project(ParticleCloudModern)

set(FFGL_SDK_DIR "${CMAKE_SOURCE_DIR}/ffgl")

add_subdirectory("${FFGL_SDK_DIR}" ffgl_build)

add_library(ParticleCloudModern SHARED
    ParticleCloudModern/ParticleCloudModern.cpp
)

target_include_directories(ParticleCloudModern PRIVATE
    "${FFGL_SDK_DIR}/source"
    "${FFGL_SDK_DIR}/source/lib"
)

target_link_libraries(ParticleCloudModern PRIVATE ffgl)

set_target_properties(ParticleCloudModern PROPERTIES
    BUNDLE TRUE
    MACOSX_BUNDLE TRUE
    OUTPUT_NAME "ParticleCloudModern"
)
