include_directories(include)
include_directories(../groupsock/include)
include_directories(../UsageEnvironment/include)
include_directories(../BasicUsageEnvironment/include)
ADD_DEFINITIONS(-D LOCALE_NOT_USED)
file(GLOB liveMediaFiles
      *.cpp *.c)
list(APPEND LM_Dependence_SRC_LIST
            ../UsageEnvironment/HashTable.cpp
            ../UsageEnvironment/strDup.cpp
            ../groupsock/GroupsockHelper.cpp
            ../groupsock/Groupsock.cpp
            ../groupsock/inet.c
            ../groupsock/NetAddress.cpp
            ../groupsock/NetInterface.cpp
            ../groupsock/GroupEId.cpp
            ../BasicUsageEnvironment/BasicHashTable.cpp
        )
add_library(liveMedia SHARED ${liveMediaFiles} ${LM_Dependence_SRC_LIST})

LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/libs/${ANDROID_ABI})

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log )

target_link_libraries(liveMedia ${log-lib})