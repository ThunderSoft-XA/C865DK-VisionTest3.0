//
// Created by Fly on 2020/2/12.
//

#ifndef _MYRTSPCLIENT_H_
#define _MYRTSPCLIENT_H_

#include <vector>
#include <string>

#define MAX_NUMBER_OF_CLIENT    (30)

// data struct definitions:
typedef struct {
    u_int8_t    clientId;
    u_int32_t   frameSize;  // size of frameData (the whole frame)
    int64_t     time; //(1000000*presentationTime.tv_sec+presentationTime.tv_usec)&0x00000000ffffffff
    std::string url;
    std::string frameType;
    std::vector<u_int8_t>  frameData;
    u_int8_t    nalu_type;
} frameInfo;

struct audioInfo {
    u_int8_t clientId;
    std::string   url;
    std::string   frameType;
    u_int32_t frameSize;
    std::vector<u_int8_t>  frameData;
};

typedef void (sendDataCallback) (frameInfo *info);

int rtspClientOpen(char const* rtspUrl, int16_t clientId, sendDataCallback *callback);
int rtspClientClose(int16_t clientId);

#endif  // _MYRTSPCLIENT_H_