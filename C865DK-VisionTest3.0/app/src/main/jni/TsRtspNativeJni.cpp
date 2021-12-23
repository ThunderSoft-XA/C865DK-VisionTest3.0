//
// Created by Fly on 2020/2/12.
//


#include <pthread.h>
#include <string.h>
#include <android/log.h>

#include "com_thundercomm_rtsp_TsRtspNativeJni.h"
#include "TsRtspNative.hh"
#include "jni.h"
#include "../cpp/RtspClient/include/TsRtspNative.hh"

#define  LOG_TAG    "nativeprint"
#define LOGE(fmt, ...) \
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "[%s]-->[%d]--> " fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)

static const char* const kClass_RenderActivity ="limedia/sdk/api/realtime/TSRTSPNative";
static JavaVM *g_JavaVM;
jobject g_object[MAX_NUMBER_OF_CLIENT];
jclass g_objectClass[MAX_NUMBER_OF_CLIENT];
jmethodID g_videoMethodID[MAX_NUMBER_OF_CLIENT];
jmethodID g_audioMethodID[MAX_NUMBER_OF_CLIENT];


jint JNI_OnLoad(JavaVM* vm, void* /* reserved */) {
    JNIEnv* env = NULL;
    g_JavaVM = vm;
    jint result = -1;

    do {
        if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_4) != JNI_OK || env == NULL) {
            break;;
        }

        /* success -- return valid version number */
        result = JNI_VERSION_1_4;
    } while (0);

    return result;
}

static JNIEnv *GetEnv() {
    int status;
    JNIEnv *envnow = NULL;

    status = g_JavaVM->GetEnv(reinterpret_cast<void **>(&envnow), JNI_VERSION_1_4);
    if (status < 0) {
        status = g_JavaVM->AttachCurrentThread(&envnow, NULL);
        if (status < 0) {
            return NULL;
        }
    }

    return envnow;
}

void sendDataToDecoder(frameInfo *pInfo) {
    if (NULL == pInfo) {
        return;
    }
//    if (pInfo->frameType == "Err") {
//        LOGE("Receive network is timeout information.\n");
//        return;
//    }
    do {
        JNIEnv *env = NULL;
        int64_t presentationTimeUs = pInfo->time;
        u_int32_t sumByte = pInfo->frameData.size();
        u_int32_t clientId = pInfo->clientId;
        u_int32_t nalu_type = pInfo->nalu_type;

        LOGE("clientId = %d, ->streamType = %s\n", clientId, pInfo->frameType.c_str());
        if (sumByte != pInfo->frameSize) {
            break;
        }

        env = GetEnv();
        jbyteArray jarrRV = env->NewByteArray(sumByte);
        jbyte *jby = env->GetByteArrayElements(jarrRV, 0);
        memcpy(jby, &(pInfo->frameData[0]), sumByte);
        env->SetByteArrayRegion(jarrRV, 0, sumByte, jby);
        jstring type = env->NewStringUTF(pInfo->frameType.c_str());

        if ((pInfo->frameType == "H264") || (pInfo->frameType == "H265")
                                          || (pInfo->frameType == "Err")) {
            env->CallVoidMethod(g_object[clientId], g_videoMethodID[clientId], \
                    clientId, nalu_type, sumByte, jarrRV, presentationTimeUs, type);
//            LOGE("Video JNI Call success ---> clientId = %d\n",clientId);
        } else {
            env->CallVoidMethod(g_object[clientId], g_audioMethodID[clientId], \
                    clientId, sumByte, jarrRV, type, presentationTimeUs);
//            LOGE("Audio JNI Call success ---> clientId = %d\n", clientId);
        }

        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }

        env->ReleaseByteArrayElements(jarrRV, jby, 0);
        env->DeleteLocalRef(jarrRV);
        env->DeleteLocalRef(type);
    } while(0);
}

/*
 * Class:     com_example_fly_live555_myRtspClientJni
 * Method:    rtspClientOpenJni
 * Signature: (Ljava/lang/String;I)I
 */
JNIEXPORT jint JNICALL Java_com_thundercomm_rtsp_TsRtspNative_rtspClientOpenJni
        (JNIEnv *env, jobject thiz, jstring url, jint clientId) {
    char* myUrl = const_cast<char*>(env->GetStringUTFChars(url, 0));
    int16_t myClientId = clientId;
    sendDataCallback* callback = sendDataToDecoder;

    LOGE("myUrl=%s, myClientId=%d \n", myUrl, myClientId);

    jint ret = rtspClientOpen(myUrl, myClientId, callback);
    env->ReleaseStringUTFChars(url, myUrl);

    if (env->ExceptionOccurred()) {
        env->ExceptionDescribe();
    }

    g_object[myClientId] = env->NewGlobalRef(thiz);

    if (env->ExceptionOccurred()) {
        env->ExceptionDescribe();
    }

    g_objectClass[myClientId] = env->GetObjectClass(thiz);
    g_videoMethodID[myClientId] = env->GetMethodID(g_objectClass[myClientId], "postDataFromNative", "(III[BJLjava/lang/String;)V");
    g_audioMethodID[myClientId] = \
            env->GetMethodID(g_objectClass[myClientId], "postAudioDataFromNative", "(II[BLjava/lang/String;J)V");

    if (env->ExceptionOccurred()) {
        env->ExceptionDescribe();
    }

    return ret;
}

/*
 * Class:     com_example_fly_live555_myRtspClientJni
 * Method:    rtspClientCloseJni
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_com_thundercomm_rtsp_TsRtspNative_rtspClientCloseJni
        (JNIEnv *env, jclass, jint clientId) {
    int16_t myClientId = clientId;
    LOGE("rtspClientCloseJni-->myClientId=%d\n", myClientId);

    jint ret = rtspClientClose(myClientId);

    return ret;
}
