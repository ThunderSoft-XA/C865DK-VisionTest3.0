include_directories(include)
include_directories(../liveMedia/include)
include_directories(../BasicUsageEnvironment/include)
include_directories(../UsageEnvironment/include)
include_directories(../groupsock/include)

file(GLOB RtspClientFiles
   *.cpp)
list(APPEND RC_Dependence_SRC_LIST
        ../../jni/TsRtspNativeJni.cpp
#            ../liveMedia/MediaSink.cpp
#			../liveMedia/MediaSession.cpp
#			../liveMedia/Media.cpp
#			../BasicUsageEnvironment/BasicUsageEnvironment.cpp
#			../BasicUsageEnvironment/BasicTaskScheduler.cpp
        )

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library(RtspClient SHARED ${RtspClientFiles} ${RC_Dependence_SRC_LIST})
#add_library(RtspClient SHARED ${RtspClientFiles})

LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/libs/${ANDROID_ABI})

#add_library( # Sets the name of the library.
#             native-lib

             # Sets the library as a shared library.
#             SHARED

             # Provides a relative path to your source file(s).
#             src/main/cpp/native-lib.cpp )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

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

target_link_libraries(RtspClient ${log-lib} BasicUsageEnvironment groupsock liveMedia UsageEnvironment)