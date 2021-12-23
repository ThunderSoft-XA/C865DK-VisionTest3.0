file(GLOB BitmapFiles
   *.cpp)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library(bitmaputil SHARED ${BitmapFiles})
#add_library(RtspClient SHARED ${RtspClientFiles})

LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/libs/${ANDROID_ABI})

#find_library( # Sets the name of the path variable.
#              ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}

              # Specifies the name of the NDK library that
              # you want CMake to locate.
#              BasicUsageEnvironment groupsock liveMedia UsageEnvironment)

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries(bitmaputil ${log-lib} jnigraphics)