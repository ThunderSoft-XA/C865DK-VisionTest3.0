//
// Created by user on 20-5-29.
//

#include <jni.h>

#include <android/log.h>
#include <android/bitmap.h>
#include <cstring>
#include <bits/sysconf.h>
#include <sched.h>

#define  LOG_TAG    "nativeprint"
#define  LOGD(fmt, ...) \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "[%s]-->[%d]--> " fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define  LOGE(fmt, ...) \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "[%s]-->[%d]--> " fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)

extern "C"
JNIEXPORT jint JNICALL
Java_com_thundercomm_eBox_Jni_00024Affinity_getCore(JNIEnv *env, jclass clazz) {
    return sysconf(_SC_NPROCESSORS_CONF);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_thundercomm_eBox_Jni_00024Affinity_bindToCpu(JNIEnv *env, jclass clazz, jint cpu) {
    int cores = sysconf(_SC_NPROCESSORS_CONF);
    LOGD("get cpu number = %d\n", cores);
    if (cpu >= cores) {
        LOGE("your set cpu is beyond the cores,exit...");
        return;
    }

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//设置线程CPU亲和力
    {
        LOGD("warning: could not set CPU affinity, continuing...\n");
    } else {
        LOGD("set affinity to %d success", cpu);
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_thundercomm_eBox_Jni_00024Affinity_uubindFromCpu(JNIEnv *env, jclass clazz, jint cpu) {
    // TODO: implement uubindFromCpu()
    int cores = sysconf(_SC_NPROCESSORS_CONF);
    LOGD("get cpu number = %d\n", cores);
    if (cpu >= cores) {
        LOGE("your set cpu is beyond the cores,exit...");
        return;
    }
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_CLR(cpu, &mask);

    if (!CPU_ISSET(cpu,&mask)){
        LOGD("unbind from CPU %d success",cpu);
    } else{
        LOGD("warning: could not unbind CPU affinity, continuing...\n");
    }
//    if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//设置线程CPU亲和力
//    {
//        LOGD("warning: could not set CPU affinity, continuing...\n");
//    } else {
//        LOGD("set affinity to %d success", cpu);
//    }
}

