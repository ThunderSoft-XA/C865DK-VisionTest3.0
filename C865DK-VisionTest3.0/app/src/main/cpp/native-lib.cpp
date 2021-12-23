//
// Created by user on 20-4-1.
//
#include "jni.h"
#include "string"
#include "stdio.h"
#include "android/log.h"
#include "android/native_window_jni.h"
#include "unistd.h"





#define TAG "JNI_TAG"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG ,__VA_ARGS__) // 定义LOGD类型
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG ,__VA_ARGS__) // 定义LOGI类型
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,TAG ,__VA_ARGS__) // 定义LOGW类型
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,TAG ,__VA_ARGS__) // 定义LOGE类型
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL,TAG ,__VA_ARGS__) // 定义LOGF类型

#define srcWidth 1920
#define srcHeight 1080

#define outWidth 1920
#define outHeight 1080

extern "C"
JNIEXPORT jint JNICALL
Java_com_thundersoft_eBox_Jni_test(JNIEnv *env, jobject thiz) {
    LOGD("Test from JNI");
    return 1;
    // TODO: implement test()
}
extern "C"
JNIEXPORT void JNICALL
Java_com_thundersoft_eBox_Jni_start_1preview(JNIEnv *env, jobject thiz, jobject surface_view) {
    // TODO: implement start_preview()
}