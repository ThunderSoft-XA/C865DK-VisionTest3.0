file(GLOB AffinityC
   *.cpp)


add_library(affinitylib SHARED src/main/cpp/Affinity/AffinityC.cpp)


LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/libs/${ANDROID_ABI})


find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )
