cmake_minimum_required(VERSION 3.15)
project(ParticleCloudModern)

set(FFGL_SDK_DIR "${CMAKE_SOURCE_DIR}/ffgl")

# Build the FFGL SDK
add_subdirectory("${FFGL_SDK_DIR}" ffgl_build)

# Create plugin library
add_library(ParticleCloudModern MODULE
    ParticleCloudModern/ParticleCloudModern.cpp
)

# Include paths so FFGL.h and others are found
target_include_directories(ParticleCloudModern PRIVATE
    "${FFGL_SDK_DIR}/source"
    "${FFGL_SDK_DIR}/source/lib"
    "${FFGL_SDK_DIR}/source/lib/ffgl"
    "${FFGL_SDK_DIR}/source/lib/ffglex"
    "${FFGL_SDK_DIR}/source/lib/ffglquickstart"
)

# Link against FFGL
target_link_libraries(ParticleCloudModern PRIVATE ffgl::sdk)


# Produce a macOS .bundle
set_target_properties(ParticleCloudModern PROPERTIES
    BUNDLE TRUE
    MACOSX_BUNDLE TRUE
    OUTPUT_NAME "ParticleCloudModern"
)
target_link_libraries(ParticleCloudModern PRIVATE "-framework OpenGL")
