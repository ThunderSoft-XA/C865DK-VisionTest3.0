/**********
This library is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version. (See <http://www.gnu.org/copyleft/lesser.html>.)

This library is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this library; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
**********/
// Copyright (c) 1996-2019, Live Networks, Inc.  All rights reserved
// A demo application, showing how to create and run a RTSP client
// (that can potentially receive multiple streams concurrently).
//
// NOTE: This code - although it builds a running application -
// is intended only to illustrate how to develop your own RTSP client application.
// For a full-featured RTSP client application - with much more functionality, and many options
// - see "openRTSP": http://www.live555.com/openRTSP/

#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <android/log.h>
#include <sys/syscall.h>


#include <vector>
#include <string>
#include <iostream>


#include "liveMedia.hh"
#include "BasicUsageEnvironment.hh"
#include "TsRtspNative.hh"


#define  LOG_TAG    "nativeprint"
#define  LOGD(fmt, ...) \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "[%s]-->[%d]--> " fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define NETWORK_TIMEOUT         (10)
#define MAX_NUMBER_OF_CLIENT    (30)

#define CLIENT_STATUS_STOP      (0)
#define CLIENT_STATUS_NORMAL    (1)
#define CLIENT_STAUS_ERROR      (2)

// define the return value macro
#define RET_OK (0)
#define RET_ERROR (-1)

// by default, print verbose output from each "RTSPClient"
#define RTSP_CLIENT_VERBOSITY_LEVEL 1

// By default, we request that the server stream its data using RTP/UDP.
// If, instead, you want to request that the server stream via RTP-over-TCP, change the following to True:
#define REQUEST_STREAMING_OVER_TCP true

// Even though we're not going to be doing anything with the incoming data, we still need to receive it.
// Define the size of the buffer that we'll use:
#define DUMMY_SINK_RECEIVE_BUFFER_SIZE 5000000
#define SAVE_TO_FILE 0
// If you don't want to see debugging output for each received frame, then comment out the following line:
#define DEBUG_PRINT_EACH_RECEIVED_FRAME 0

// Counts how many streams (i.e., "RTSPClient"s) are currently in use.
static unsigned rtspClientCount = 0;
char eventLoopWatchVariable = 0;


// Forward function definitions:

// RTSP 'response handlers':
void continueAfterDESCRIBE(RTSPClient *rtspClient, int resultCode, char *resultString);

void continueAfterSETUP(RTSPClient *rtspClient, int resultCode, char *resultString);

void continueAfterPLAY(RTSPClient *rtspClient, int resultCode, char *resultString);

// Other event handler functions:
// called when a stream's subsession (e.g., audio or video substream) ends
void subsessionAfterPlaying(void *clientData);

void subsessionByeHandler(void *clientData, char const *reason);

// called when a RTCP "BYE" is received for a subsession
void streamTimerHandler(void *clientData);
// called at the end of a stream's expected duration (if the stream has not already signaled its end using a RTCP "BYE")

// Used to iterate through each stream's 'subsessions', setting up each one:
void setupNextSubsession(RTSPClient *rtspClient);

// Used to shut down and close a stream (including its "RTSPClient" object):
void shutdownStream(RTSPClient *rtspClient, int exitCode = 1);

// A function that outputs a string that identifies each stream (for debugging output).  Modify this if you wish:
UsageEnvironment &operator<<(UsageEnvironment &env, const RTSPClient &rtspClient) {
    return env << "[URL:\"" << rtspClient.url() << "\"]: ";
}

// A function that outputs a string that identifies each subsession (for debugging output).  Modify this if you wish:
UsageEnvironment &operator<<(UsageEnvironment &env, const MediaSubsession &subsession) {
    return env << subsession.mediumName() << "/" << subsession.codecName();
}

void usage(UsageEnvironment &env, char const *progName) {
    env << "Usage: " << progName << " <rtsp-url-1> ... <rtsp-url-N>\n";
    env << "\t(where each <rtsp-url-i> is a \"rtsp://\" URL)\n";
}

// Define a class to hold per-stream state that we maintain throughout each stream's lifetime:

class StreamClientState {
public:
    StreamClientState();

    virtual ~StreamClientState();

public:
    MediaSubsessionIterator *iter;
    MediaSession *session;
    MediaSubsession *subsession;
    TaskToken streamTimerTask;
    double duration;
};

// If you're streaming just a single stream (i.e., just from a single URL, once), then you can define and use
// just a single "StreamClientState" structure, as a global variable in your application.
// However, because - in this demo application - we're showing how to play multiple streams, concurrently,
// we can't do that.  Instead, we have to have a separate "StreamClientState" structure for each "RTSPClient".
// To do this, we subclass "RTSPClient", and add a "StreamClientState" field to the subclass:

class ourRTSPClient : public RTSPClient {
public:
    static ourRTSPClient *createNew(UsageEnvironment &env, char const *rtspURL,
                                    int verbosityLevel = 0,
                                    char const *applicationName = NULL,
                                    portNumBits tunnelOverHTTPPortNum = 0);

protected:
    ourRTSPClient(UsageEnvironment &env, char const *rtspURL,
                  int verbosityLevel, char const *applicationName,
                  portNumBits tunnelOverHTTPPortNum);

    // called only by createNew();
    virtual ~ourRTSPClient();

public:
    StreamClientState scs;
};

// Define a data sink (a subclass of "MediaSink") to receive the data for each subsession
// (i.e., each audio or video 'substream').
// In practice, this might be a class (or a chain of classes) that decodes and then renders the incoming audio or video.
// Or it might be a "FileSink", for outputting the received data into a file (as is done by the "openRTSP" application).
// In this example code, however, we define a simple 'dummy' sink that receives incoming data, but does nothing with it.

class DummySink : public MediaSink {
public:
    static DummySink *createNew(UsageEnvironment &env,
                                MediaSubsession &subsession,    // identifies the kind of data that's being received
                                char const *streamId = NULL);   // identifies the stream itself (optional)
    void setOrbUrl(char const* url);
    char const* orgurl() const { return orgfStreamId; }
private:
    DummySink(UsageEnvironment &env, MediaSubsession &subsession, char const *streamId);

    // called only by "createNew()"
    virtual ~DummySink();

    static void afterGettingFrame(void *clientData, unsigned frameSize,
                                  unsigned numTruncatedBytes,
                                  struct timeval presentationTime,
                                  unsigned durationInMicroseconds);

    void afterGettingFrame(unsigned frameSize, unsigned numTruncatedBytes,
                           struct timeval presentationTime, unsigned durationInMicroseconds);

private:
    // redefined virtual functions:
    virtual Boolean continuePlaying();

private:
    u_int8_t *fReceiveBuffer;
    MediaSubsession &fSubsession;
    char *fStreamId;
    char *orgfStreamId;
    Boolean isFirstFrame_ = true;
};


// data struct definitions:
typedef struct {
    int16_t clientId;       // 0 <= clientId <=3 , other value is invalid.
    int16_t clientState;    // 0:stop,  1:normal,  2:error
    sendDataCallback *callback;
    char monitorFlag;
    std::string urlData;
    pthread_t threadPid;
    time_t curTime;
    TaskScheduler *scheduler;
    UsageEnvironment *env;
    ourRTSPClient *clientHandle;
} rtspSingleClient;

typedef struct {
    int16_t activeClientCount;
    bool initState;      // true: has been inited
    pthread_t threadPid;
    sendDataCallback *callback;
    std::vector<rtspSingleClient> rtspClientData;
} rtspClinetManger;

rtspClinetManger g_multiClientManager;

static u_int8_t getClientId(std::string rtspUrl) {
    u_int8_t i = 0;
    char *revUrl = const_cast<char *>(rtspUrl.c_str());

    //LOGD("InputUrl=%s\n", rtspUrl.c_str());

    for (i = 0; i < MAX_NUMBER_OF_CLIENT; i++) {
        //LOGD("urlData=%s\n", g_multiClientManager.rtspClientData[i].urlData.c_str());
        if (g_multiClientManager.rtspClientData[i].urlData.empty()) {
            continue;
        }
        char *urlData = const_cast<char *>(g_multiClientManager.rtspClientData[i].urlData.c_str());
        char *tmpUrlData = strstr(urlData, "@");
        char *tmpRevUrl = strstr(revUrl, "@");
        if (tmpUrlData == NULL) {
            tmpUrlData = revUrl;
        }

        if (tmpUrlData && tmpRevUrl) {
            if (!strncmp(urlData, revUrl, strlen(revUrl) - 1)) {
                //LOGD("find ClientID = %d \n", i);
                return i;
            }
        } else {
            #if 0
            // compare IP address only
            char *tmp = strstr(&tmpUrlData[1], ":");
            if (NULL == tmp) {
                tmp = strstr(&tmpUrlData[1], "/");
            }

            if (!strncmp(&tmpUrlData[1], &revUrl[strlen("rtsp://")], tmp - tmpUrlData - 1)) {
                LOGD("find ClientID = %d \n", i);
                return i;
            }
            #else
            if (!strncmp(urlData, tmpUrlData, strlen(tmpUrlData) - 1)) {
                //LOGD("find ClientID = %d \n", i);
                return i;
            }
            #endif
        }
    }
    LOGD("Can't find ClientID\n");
    return 0xFF;
}


// Implementation of the RTSP 'response handlers':

void continueAfterDESCRIBE(RTSPClient *rtspClient, int resultCode, char *resultString) {
    LOGD();
    do {
        UsageEnvironment &env = rtspClient->envir();    // alias
        StreamClientState &scs = (reinterpret_cast<ourRTSPClient *>(rtspClient))->scs;     // alias

        if (resultCode != 0) {
            env << *rtspClient << "Failed to get a SDP description: " << resultString << "\n" << "resultCode" << resultCode;
            delete[] resultString;
            LOGD("Failed to get a SDP description: resultCode", resultCode);
            break;
        }

        char *const sdpDescription = resultString;
        env << *rtspClient << "Got a SDP description:\n" << sdpDescription << "\n";
        LOGD("Got a SDP description:\n %s", sdpDescription);

        // Create a media session object from this SDP description:
        scs.session = MediaSession::createNew(env, sdpDescription);
        delete[] sdpDescription;    // because we don't need it anymore
        if (scs.session == NULL) {
            env << *rtspClient
                << "Failed to create a MediaSession object from the SDP description: " \
 << env.getResultMsg() << "\n";
            LOGD("Failed to create a MediaSession object from the SDP description: ");
            break;
        } else if (!scs.session->hasSubsessions()) {
            env << *rtspClient << "This session has no media subsessions (i.e., no \"m=\" lines)\n";
            LOGD("This session has no media subsessions (i.e., no \"m=\" lines)\n");
            break;
        }

        // Then, create and set up our data source objects for the session.
        // We do this by iterating over the session's 'subsessions',
        // calling "MediaSubsession::initiate()", and then sending a RTSP "SETUP" command, on each one.
        // (Each 'subsession' will have its own data source.)
        scs.iter = new MediaSubsessionIterator(*scs.session);
        setupNextSubsession(rtspClient);
        return;
    } while (0);

    // An unrecoverable error occurred with this stream.
    shutdownStream(rtspClient);
}


void setupNextSubsession(RTSPClient *rtspClient) {
    UsageEnvironment &env = rtspClient->envir();    // alias
    StreamClientState &scs = (reinterpret_cast<ourRTSPClient *>(rtspClient))->scs;     // alias
    LOGD();

    scs.subsession = scs.iter->next();
    if (scs.subsession != NULL) {
        if (!scs.subsession->initiate()) {
            env << *rtspClient << "Failed to initiate the \"" << *scs.subsession
                << "\" subsession: " \
 << env.getResultMsg() << "\n";
            setupNextSubsession(rtspClient);    // give up on this subsession; go to the next one
        } else {
            env << *rtspClient << "Initiated the \"" << *scs.subsession << "\" subsession (";
            if (scs.subsession->rtcpIsMuxed()) {
                env << "client port " << scs.subsession->clientPortNum();
            } else {
                env << "client ports " << scs.subsession->clientPortNum() << "-"
                    << scs.subsession->clientPortNum() + 1;
            }
            env << ")\n";

            // Continue setting up this subsession, by sending a RTSP "SETUP" command:
            rtspClient->sendSetupCommand(*scs.subsession, continueAfterSETUP, False,
                                         REQUEST_STREAMING_OVER_TCP);
        }
        return;
    }

    // We've finished setting up all of the subsessions.  Now, send a RTSP "PLAY" command to start the streaming:
    if (scs.session->absStartTime() != NULL) {
        // Special case: The stream is indexed by 'absolute' time, so send an appropriate "PLAY" command:
        rtspClient->sendPlayCommand(*scs.session, continueAfterPLAY, scs.session->absStartTime(), \
                                    scs.session->absEndTime());
    } else {
        scs.duration = scs.session->playEndTime() - scs.session->playStartTime();
        rtspClient->sendPlayCommand(*scs.session, continueAfterPLAY);
    }
}

void continueAfterSETUP(RTSPClient *rtspClient, int resultCode, char *resultString) {
    do {
        UsageEnvironment &env = rtspClient->envir();    // alias
        StreamClientState &scs = (reinterpret_cast<ourRTSPClient *>(rtspClient))->scs;     // alias
        LOGD();

        if (resultCode != 0) {
            env << *rtspClient << "Failed to set up the \"" << *scs.subsession << "\" subsession: " \
 << resultString << "\n";
            break;
        }

        env << *rtspClient << "Set up the \"" << *scs.subsession << "\" subsession (";
        if (scs.subsession->rtcpIsMuxed()) {
            env << "client port " << scs.subsession->clientPortNum();
        } else {
            env << "client ports " << scs.subsession->clientPortNum() << "-"
                << scs.subsession->clientPortNum() + 1;
        }
        env << ")\n";

        // Having successfully setup the subsession, create a data sink for it, and call "startPlaying()" on it.
        // (This will prepare the data sink to receive data;
        // the actual flow of data from the client won't start happening until later,
        // after we've sent a RTSP "PLAY" command.)
        DummySink* dummysink = DummySink::createNew(env, *scs.subsession, rtspClient->url());
        dummysink->setOrbUrl(rtspClient->orgurl());
        scs.subsession->sink = dummysink;
        // perhaps use your own custom "MediaSink" subclass instead
        if (scs.subsession->sink == NULL) {
            env << *rtspClient << "Failed to create a data sink for the \"" << *scs.subsession
                << "\" subsession: " << env.getResultMsg() << "\n";
            break;
        }

        env << *rtspClient << "Created a data sink for the \"" << *scs.subsession
            << "\" subsession\n";
        // a hack to let subsession handler functions get the "RTSPClient" from the subsession
        scs.subsession->miscPtr = rtspClient;
        scs.subsession->sink->startPlaying(*(scs.subsession->readSource()),
                                           subsessionAfterPlaying, scs.subsession);
        // Also set a handler to be called if a RTCP "BYE" arrives for this subsession:
        if (scs.subsession->rtcpInstance() != NULL) {
            scs.subsession->rtcpInstance()->setByeWithReasonHandler(subsessionByeHandler,
                                                                    scs.subsession);
        }
    } while (0);
    delete[] resultString;

    // Set up the next subsession, if any:
    setupNextSubsession(rtspClient);
}

void continueAfterPLAY(RTSPClient *rtspClient, int resultCode, char *resultString) {
    Boolean success = False;
    LOGD();

    do {
        UsageEnvironment &env = rtspClient->envir();    // alias
        StreamClientState &scs = (reinterpret_cast<ourRTSPClient *>(rtspClient))->scs;   // alias

        if (resultCode != 0) {
            env << *rtspClient << "Failed to start playing session: " << resultString << "\n";
            break;
        }

        // Set a timer to be handled at the end of the stream's expected duration
        // (if the stream does not already signal its end using a RTCP "BYE").
        // This is optional.  If, instead, you want to keep the stream active - e.g., so you can later
        // 'seek' back within it and do another RTSP "PLAY" - then you can omit this code.
        // (Alternatively, if you don't want to receive the entire stream,
        // you could set this timer for some shorter value.)
        if (scs.duration > 0) {
            // number of seconds extra to delay, after the stream's expected duration.  (This is optional.)
            unsigned const delaySlop = 2;
            scs.duration += delaySlop;
            unsigned uSecsToDelay = (unsigned) (scs.duration * 1000000);
            scs.streamTimerTask = env.taskScheduler().scheduleDelayedTask(uSecsToDelay, \
                    reinterpret_cast<TaskFunc *>(streamTimerHandler), rtspClient);
        }

        env << *rtspClient << "Started playing session";
        if (scs.duration > 0) {
            env << " (for up to " << scs.duration << " seconds)";
        }
        env << "...\n";

        success = True;
    } while (0);
    delete[] resultString;

    if (!success) {
        // An unrecoverable error occurred with this stream.
        shutdownStream(rtspClient);
    }
}


// Implementation of the other event handlers:
void subsessionAfterPlaying(void *clientData) {
    MediaSubsession *subsession = reinterpret_cast<MediaSubsession *>(clientData);
    RTSPClient *rtspClient = reinterpret_cast<RTSPClient *>(subsession->miscPtr);
    LOGD();

    // Begin by closing this subsession's stream:
    Medium::close(subsession->sink);
    subsession->sink = NULL;

    // Next, check whether *all* subsessions' streams have now been closed:
    MediaSession &session = subsession->parentSession();
    MediaSubsessionIterator iter(session);
    while ((subsession = iter.next()) != NULL) {
        if (subsession->sink != NULL) {
            return;     // this subsession is still active
        }
    }

    // All subsessions' streams have now been closed, so shutdown the client:
    shutdownStream(rtspClient);
}

void subsessionByeHandler(void *clientData, char const *reason) {
    MediaSubsession *subsession = reinterpret_cast<MediaSubsession *>(clientData);
    RTSPClient *rtspClient = reinterpret_cast<RTSPClient *>(subsession->miscPtr);
    UsageEnvironment &env = rtspClient->envir();    // alias
    LOGD();

    env << *rtspClient << "Received RTCP \"BYE\"";
    if (reason != NULL) {
        env << " (reason:\"" << reason << "\")";
        delete[] reason;
    }
    env << " on \"" << *subsession << "\" subsession\n";

    // Now act as if the subsession had closed:
    subsessionAfterPlaying(subsession);
}

void streamTimerHandler(void *clientData) {
    ourRTSPClient *rtspClient = reinterpret_cast<ourRTSPClient *>(clientData);
    StreamClientState &scs = rtspClient->scs;   // alias
    LOGD();

    scs.streamTimerTask = NULL;

    // Shut down the stream:
    shutdownStream(rtspClient);
}

void shutdownStream(RTSPClient *rtspClient, int exitCode) {
    UsageEnvironment &env = rtspClient->envir();    // alias
    StreamClientState &scs = (reinterpret_cast<ourRTSPClient *>(rtspClient))->scs;   // alias
    LOGD();

    // First, check whether any subsessions have still to be closed:
    if (scs.session != NULL) {
        Boolean someSubsessionsWereActive = False;
        MediaSubsessionIterator iter(*scs.session);
        MediaSubsession *subsession;

        while ((subsession = iter.next()) != NULL) {
            if (subsession->sink != NULL) {
                Medium::close(subsession->sink);
                subsession->sink = NULL;

                if (subsession->rtcpInstance() != NULL) {
                    // in case the server sends a RTCP "BYE" while handling "TEARDOWN"
                    subsession->rtcpInstance()->setByeHandler(NULL, NULL);
                }

                someSubsessionsWereActive = True;
            }
        }

        if (someSubsessionsWereActive) {
            // Send a RTSP "TEARDOWN" command, to tell the server to shutdown the stream.
            // Don't bother handling the response to the "TEARDOWN".
            rtspClient->sendTeardownCommand(*scs.session, NULL);
        }
    }

    env << *rtspClient << "Closing the stream.\n";
    Medium::close(rtspClient);
    // Note that this will also cause this stream's "StreamClientState" structure to get reclaimed.

    if (--rtspClientCount == 0) {
        // The final stream has ended, so exit the application now.
        // (Of course, if you're embedding this code into your own application, you might want to comment this out,
        // and replace it with "eventLoopWatchVariable = 1;",
        // so that we leave the LIVE555 event loop, and continue running "main()".)
        exit(exitCode);
    }
}


// Implementation of "ourRTSPClient":
ourRTSPClient *ourRTSPClient::createNew(UsageEnvironment &env, char const *rtspURL,
                                        int verbosityLevel, char const *applicationName,
                                        portNumBits tunnelOverHTTPPortNum) {
    return new ourRTSPClient(env, rtspURL, verbosityLevel, applicationName, tunnelOverHTTPPortNum);
}

ourRTSPClient::ourRTSPClient(UsageEnvironment &env, char const *rtspURL,
                             int verbosityLevel, char const *applicationName,
                             portNumBits tunnelOverHTTPPortNum)
        : RTSPClient(env, rtspURL, verbosityLevel, applicationName, tunnelOverHTTPPortNum, -1) {
}

ourRTSPClient::~ourRTSPClient() {
}


// Implementation of "StreamClientState":
StreamClientState::StreamClientState()
        : iter(NULL), session(NULL), subsession(NULL), streamTimerTask(NULL), duration(0.0) {
}

StreamClientState::~StreamClientState() {
    delete iter;
    if (session != NULL) {
        // We also need to delete "session", and unschedule "streamTimerTask" (if set)
        UsageEnvironment &env = session->envir();   // alias

        env.taskScheduler().unscheduleDelayedTask(streamTimerTask);
        Medium::close(session);
    }
}


// Implementation of "DummySink":
DummySink *
DummySink::createNew(UsageEnvironment &env, MediaSubsession &subsession, char const *streamId) {
    return new DummySink(env, subsession, streamId);
}

void DummySink::setOrbUrl(char const* url) {
  delete[] orgfStreamId; orgfStreamId = strDup(url);
}

DummySink::DummySink(UsageEnvironment &env, MediaSubsession &subsession, char const *streamId)
        : MediaSink(env),
          fSubsession(subsession), orgfStreamId(NULL) {

    fStreamId = strDup(streamId);
    fReceiveBuffer = new u_int8_t[DUMMY_SINK_RECEIVE_BUFFER_SIZE];
}

DummySink::~DummySink() {
    delete[] fReceiveBuffer;
    delete[] fStreamId;
    delete[] orgfStreamId;
}

void DummySink::afterGettingFrame(void *clientData, unsigned frameSize, unsigned numTruncatedBytes,
                                  struct timeval presentationTime,
                                  unsigned durationInMicroseconds) {
    DummySink *sink = reinterpret_cast<DummySink *>(clientData);
    sink->afterGettingFrame(frameSize, numTruncatedBytes, presentationTime, durationInMicroseconds);
}

void DummySink::afterGettingFrame(unsigned frameSize, unsigned numTruncatedBytes,
                                  struct timeval presentationTime,
                                  unsigned /*durationInMicroseconds*/) {
    u_int32_t i = 0;

    // We've just received a frame of data.  (Optionally) print out information about it:
    if (fStreamId != NULL) {
        //LOGD("Stream [%s --> %s --> %s] :\tReceived %d bytes\n", \
        //     fStreamId, fSubsession.mediumName(), fSubsession.codecName(), frameSize);
    }

    // send video frame to app decoder
    if (fReceiveBuffer
        && (!strcmp(fSubsession.mediumName(), "video"))
        && ((0 == strncmp(fSubsession.codecName(), "H264", 16)) ||
            (0 == strncmp(fSubsession.codecName(), "H265", 16)))) {
        unsigned char nalu_header[4] = {0, 0, 0, 1};
        unsigned char extraData[256];
        unsigned int extraLen = 0;

        if (0 == strncmp(fSubsession.codecName(), "H264", 16)) {
            SPropRecord *pSPropRecord;
            unsigned int num = 0;
            pSPropRecord = parseSPropParameterSets(fSubsession.fmtp_spropparametersets(), num);
            // p_record[0] is sps
            // p_record[1] is pps
            for (i = 0; i < num; i++) {
                memcpy(&extraData[extraLen], &nalu_header[0], 4);
                extraLen += 4;
                memcpy(&extraData[extraLen], pSPropRecord[i].sPropBytes, pSPropRecord[i].sPropLength);
                extraLen += pSPropRecord[i].sPropLength;
            }
            if (pSPropRecord) delete[] pSPropRecord;
        } else if (0 == strncmp(fSubsession.codecName(), "H265", 16)) {
            SPropRecord* sPropRecords[3];
            unsigned numSPropRecords[3];
            sPropRecords[0] = parseSPropParameterSets(fSubsession.fmtp_spropvps(), numSPropRecords[0]);
            sPropRecords[1] = parseSPropParameterSets(fSubsession.fmtp_spropsps(), numSPropRecords[1]);
            sPropRecords[2] = parseSPropParameterSets(fSubsession.fmtp_sproppps(), numSPropRecords[2]);
            for (i = 0; i < numSPropRecords[0]; i++) {
                if (sPropRecords[0][i].sPropLength > 0) {
                    memcpy(&extraData[extraLen], &nalu_header[0], 4);
                    extraLen += 4;
                    memcpy(&extraData[extraLen], sPropRecords[0][i].sPropBytes, sPropRecords[0][i].sPropLength);
                    extraLen += sPropRecords[0][i].sPropLength;
                }
            }
            for (i = 0; i < numSPropRecords[1]; i++) {
                if (sPropRecords[1][i].sPropLength > 0) {
                    memcpy(&extraData[extraLen], &nalu_header[0], 4);
                    extraLen += 4;
                    memcpy(&extraData[extraLen], sPropRecords[1][i].sPropBytes, sPropRecords[1][i].sPropLength);
                    extraLen += sPropRecords[1][i].sPropLength;
                }
            }
            for (i = 0; i < numSPropRecords[2]; i++) {
                if (sPropRecords[2][i].sPropLength > 0) {
                    memcpy(&extraData[extraLen], &nalu_header[0], 4);
                    extraLen += 4;
                    memcpy(&extraData[extraLen], sPropRecords[2][i].sPropBytes, sPropRecords[2][i].sPropLength);
                    extraLen += sPropRecords[2][i].sPropLength;
                }
            }
            if (sPropRecords[0]) delete[] sPropRecords[0];
            if (sPropRecords[1]) delete[] sPropRecords[1];
            if (sPropRecords[2]) delete[] sPropRecords[2];
        }

        memcpy(&extraData[extraLen], &nalu_header[0], 4);
        extraLen += 4;

        frameInfo videoInfo;
        char* checkUrl = strdup(orgfStreamId);
        videoInfo.url = strdup(fStreamId);
        videoInfo.clientId = getClientId(checkUrl);  // getClientId(videoInfo.url);
        delete[] checkUrl;
        if (videoInfo.clientId >= MAX_NUMBER_OF_CLIENT) {
            goto NEXT;
        }

        g_multiClientManager.rtspClientData[videoInfo.clientId].curTime = time(NULL);
        g_multiClientManager.rtspClientData[videoInfo.clientId].clientState = CLIENT_STATUS_NORMAL;
        videoInfo.frameType = fSubsession.codecName();
        videoInfo.time =
                (1000000 * presentationTime.tv_sec + presentationTime.tv_usec) & 0x00000000ffffffff;
        videoInfo.frameSize = extraLen + frameSize;
        videoInfo.frameData.resize(videoInfo.frameSize);
        for (i = 0; i < extraLen; i++) {
            videoInfo.frameData[i] = extraData[i];
        }

        for (; i < videoInfo.frameSize; i++) {
            videoInfo.frameData[i] = fReceiveBuffer[i - extraLen];
        }

        if (0 == strncmp(fSubsession.codecName(), "H264", 16)) {
            videoInfo.nalu_type = (fReceiveBuffer[0] & 0x1F);
        } else if (0 == strncmp(fSubsession.codecName(), "H265", 16)) {
            videoInfo.nalu_type = ((fReceiveBuffer[0] & 0x7E) >> 1);
        }

        // send to JNI
        LOGD("Send video frame  to JNI, clientId = %d, RevURL = %s, cur_PID = 0x%x\n", \
             videoInfo.clientId, videoInfo.url.c_str(), (pthread_t) (pthread_self()));

        if (g_multiClientManager.rtspClientData[videoInfo.clientId].callback) {
            g_multiClientManager.rtspClientData[videoInfo.clientId].callback(&videoInfo);
        } else {
            LOGD("Callback function is NULL\n");
        }
    }

    // send audio frame to app decoder
    if (fReceiveBuffer && (!strcmp(fSubsession.mediumName(), "audio"))) {
        frameInfo audioInfo;
        audioInfo.url = strdup(fStreamId);
        audioInfo.clientId = getClientId(audioInfo.url);
        if (audioInfo.clientId >= MAX_NUMBER_OF_CLIENT) {
            goto NEXT;
        }
        g_multiClientManager.rtspClientData[audioInfo.clientId].curTime = time(NULL);
        g_multiClientManager.rtspClientData[audioInfo.clientId].clientState = CLIENT_STATUS_NORMAL;
        audioInfo.url = fStreamId;
        audioInfo.frameType = fSubsession.codecName();
        audioInfo.time =
                (1000000 * presentationTime.tv_sec + presentationTime.tv_usec) & 0x00000000ffffffff;
        audioInfo.frameSize = frameSize;
        audioInfo.frameData.resize(audioInfo.frameSize);

        for (i = 0; i < audioInfo.frameSize; i++) {
            audioInfo.frameData[i] = fReceiveBuffer[i];
        }

        // send to JNI
        if (g_multiClientManager.rtspClientData[audioInfo.clientId].callback) {
//            LOGD("Callback function is NOT NULL, then send.\n");
            g_multiClientManager.rtspClientData[audioInfo.clientId].callback(&audioInfo);
        } else {
            LOGD("Callback function is NULL\n");
        }
    }

    NEXT:
    // Then continue, to request the next frame of data:
    continuePlaying();
}

Boolean DummySink::continuePlaying() {
    if (fSource == NULL) {
        return False;   // sanity check (should not happen)
    }

    // Request the next frame of data from our input source.
    // "afterGettingFrame()" will get called later, when it arrives:
    fSource->getNextFrame(fReceiveBuffer, DUMMY_SINK_RECEIVE_BUFFER_SIZE,
                          afterGettingFrame, this, onSourceClosure, this);

    return True;
}

static void *threadFuncNetworkMonitor(void *) {
    while (g_multiClientManager.initState) {
        time_t nowTime = time(NULL);
        for (u_int8_t i = 0; i < MAX_NUMBER_OF_CLIENT; i++) {
            if (CLIENT_STATUS_NORMAL != g_multiClientManager.rtspClientData[i].clientState) {
                continue;
            }
            if (NETWORK_TIMEOUT <= (nowTime - g_multiClientManager.rtspClientData[i].curTime)) {
                // send timeout info to app
                LOGD("Timeout, clientId = %d, state = %d\n", i,
                     g_multiClientManager.rtspClientData[i].clientState);
                frameInfo errInfo;
                errInfo.clientId = i;
                errInfo.frameType = "Err";
                errInfo.frameSize = 32;
                errInfo.frameData.resize(errInfo.frameSize);
                errInfo.frameData[0] = 0;
                errInfo.frameData[1] = 0;
                errInfo.frameData[2] = 0;
                errInfo.frameData[3] = 1;
                for (u_int8_t k = 4; k < errInfo.frameSize; k++) {
                    errInfo.frameData[k] = 0xFF;
                }

                if (g_multiClientManager.callback) {
                    g_multiClientManager.callback(&errInfo);
                    g_multiClientManager.rtspClientData[i].clientState = CLIENT_STAUS_ERROR;
                } else {
                    LOGD("Callback function is NULL\n");
                }
            }
        }
        sleep(5);
    }

    LOGD("Exit the thread.\n");
    return NULL;
}


int multiClientManagerInit(sendDataCallback *callback) {
    if (!g_multiClientManager.initState) {
        g_multiClientManager.initState = true;
        g_multiClientManager.activeClientCount = 0;
        g_multiClientManager.callback = callback;
        g_multiClientManager.rtspClientData.resize(MAX_NUMBER_OF_CLIENT);
        for (int16_t i = 0; i < MAX_NUMBER_OF_CLIENT; i++) {
            g_multiClientManager.rtspClientData[i].clientId = 0xFF;
            g_multiClientManager.rtspClientData[i].clientHandle = NULL;
            g_multiClientManager.rtspClientData[i].threadPid = -1;
            g_multiClientManager.rtspClientData[i].env = NULL;
            g_multiClientManager.rtspClientData[i].scheduler = NULL;
            g_multiClientManager.rtspClientData[i].curTime = 0;
            g_multiClientManager.rtspClientData[i].clientState = CLIENT_STATUS_STOP;
        }

        if (0 != pthread_create(&(g_multiClientManager.threadPid), NULL, threadFuncNetworkMonitor,
                                NULL)) {
            LOGD("Network monitor thread create is failed.\n");
            return RET_ERROR;
        }

        pthread_detach(g_multiClientManager.threadPid);
    } else {
        LOGD("Has been inited.\n");
    }

    return RET_OK;
}

#ifndef CPU_ZERO
#define CPU_SETSIZE 1024
#define __NCPUBITS  (8 * sizeof (unsigned long))
typedef struct
{
    unsigned long __bits[CPU_SETSIZE / __NCPUBITS];
} cpu_set_t;

#define CPU_SET(cpu, cpusetp) \
  ((cpusetp)->__bits[(cpu)/__NCPUBITS] |= (1UL << ((cpu) % __NCPUBITS)))
#define CPU_ZERO(cpusetp) \
  memset((cpusetp), 0, sizeof(cpu_set_t))
#endif

void set_cur_thread_affinity(uintptr_t mask) {
    int err, syscallres;
    pid_t pid = gettid();
    syscallres = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
    if (syscallres) {
        err = errno;
        LOGD("Error in the syscall setaffinity: mask = %d, err=%d", mask, errno);
    }
    LOGD("tid = %d has setted affinity success", pid);
}

static int getCores() {
    return sysconf(_SC_NPROCESSORS_CONF);
}

static void *threadFuncSingleClient(void *pClientId) {
    uint16_t clientId = 0xFF;
    int cpu;
    cpu_set_t mask;
    int cores;

    do {
        if (pClientId) {
            clientId = *(reinterpret_cast<uint16_t *>(pClientId));
        } else {
            LOGD("pClientId == NULL\n");
            break;
        }
        LOGD("clientId = %d\n", clientId);
        cpu = clientId % 4  + 0;

        cores = getCores();
        LOGD("get cpu number = %d\n", cores);
        if (cpu >= cores) {
            LOGD("your set cpu is beyond the cores,auto to set...");
        } else {
            CPU_ZERO(&mask);
            CPU_SET(cpu, &mask);
            set_cur_thread_affinity((uintptr_t)(&mask));
            LOGD("CPU set affinity to %d success", cpu);
        }

        g_multiClientManager.rtspClientData[clientId].scheduler = BasicTaskScheduler::createNew();
        g_multiClientManager.rtspClientData[clientId].env = \
                BasicUsageEnvironment::createNew(
                *g_multiClientManager.rtspClientData[clientId].scheduler);

        ourRTSPClient *rtspClient = ourRTSPClient::createNew(
                *g_multiClientManager.rtspClientData[clientId].env, \

                const_cast<char *>(g_multiClientManager.rtspClientData[clientId].urlData.c_str()), \
                                    RTSP_CLIENT_VERBOSITY_LEVEL, NULL);
        if (rtspClient == NULL) {
            LOGD("Failed to create a RTSP client for URL =\"%s \": %s\n", \
                    g_multiClientManager.rtspClientData[clientId].urlData.c_str(), \
                    g_multiClientManager.rtspClientData[clientId].env->getResultMsg());
            break;
        }


//        LOGD("rtspClient = %p\n", rtspClient);
        g_multiClientManager.rtspClientData[clientId].clientHandle = rtspClient;
        g_multiClientManager.rtspClientData[clientId].monitorFlag = 0;
        g_multiClientManager.activeClientCount++;

        // Next, send a RTSP "DESCRIBE" command, to get a SDP description for the stream.
        // Note that this command - like all RTSP commands - is sent asynchronously;
        // we do not block, waiting for a response.
        // Instead, the following function call returns immediately,
        // and we handle the RTSP response later, from within the event loop:
        rtspClient->sendDescribeCommand(continueAfterDESCRIBE);

//        g_multiClientManager.rtspClientData[clientId].env->taskScheduler().doEventLoop(&eventLoopWatchVariable);
        g_multiClientManager.rtspClientData[clientId].env->taskScheduler().doEventLoop(\
                &g_multiClientManager.rtspClientData[clientId].monitorFlag);
    } while (0);

    LOGD("exit thread----- caihy.\n");
    return NULL;
}


int rtspClientOpen(char const *rtspUrl, int16_t clientId, sendDataCallback *callback) {
    int32_t ret = -1;
    LOGD("URL = %s, clientId = %d\n", rtspUrl, clientId);

    if (clientId >= MAX_NUMBER_OF_CLIENT) {
        LOGD("Client ID is wrong. It is bigger than 3. It shoule be in [0, 3].\n");
        return RET_ERROR;
    }

    if (NULL == callback) {
        LOGD("JNI Send callback is NULL!!!!!!\n");
        return RET_ERROR;
    } else {
        LOGD("JNI Send callback is OK!!!\n");
    }

    if (multiClientManagerInit(callback)) {
        return RET_ERROR;
    }

    do {
        if (!g_multiClientManager.initState) {
            LOGD("MultiClinet manager has not been init.\n");
            break;
        }

        if (g_multiClientManager.activeClientCount >= MAX_NUMBER_OF_CLIENT) {
            LOGD("Over the maximum number of clients\n");
            break;
        }

        g_multiClientManager.rtspClientData[clientId].clientId = clientId;
        g_multiClientManager.rtspClientData[clientId].urlData = strdup(rtspUrl);
        g_multiClientManager.rtspClientData[clientId].callback = callback;
        LOGD("Open client_ID = %d\n", g_multiClientManager.rtspClientData[clientId].clientId);
        LOGD("URLData = %s\n", g_multiClientManager.rtspClientData[clientId].urlData.c_str());
        ret = pthread_create(&(g_multiClientManager.rtspClientData[clientId].threadPid), NULL, \
                             threadFuncSingleClient, \
                             (void *) (&(g_multiClientManager.rtspClientData[clientId].clientId)));
        if (ret != 0) {
            LOGD("create client thread failed.\n");
            break;
        }

        ret = pthread_detach(g_multiClientManager.rtspClientData[clientId].threadPid);

        return RET_OK;
    } while (0);

    return RET_ERROR;
}

int rtspClientClose(int16_t clientId) {
    LOGD("clientId = %d, activeClientCount = %d \n", clientId,
         g_multiClientManager.activeClientCount);
    if (clientId >= MAX_NUMBER_OF_CLIENT) {
        LOGD("Client ID is wrong. It is bigger than 3. It shoule be in [0, 3].\n");
        return RET_ERROR;
    }

    for (int16_t i = 0; i < MAX_NUMBER_OF_CLIENT; i++) {
        if (clientId != g_multiClientManager.rtspClientData[i].clientId) {
            continue;
        }
        LOGD("clientId find\n");

        if (NULL == g_multiClientManager.rtspClientData[i].clientHandle) {
            LOGD("clientHandle is exixt\n");
            break;
        }

        StreamClientState &scs = \
                (reinterpret_cast<ourRTSPClient *>(g_multiClientManager.rtspClientData[i].clientHandle))->scs;  // alias

//        if (g_multiClientManager.rtspClientData[i].clientHandle && scs.session) {
        if (g_multiClientManager.rtspClientData[i].clientHandle) {

            LOGD("send teardown command\n");
            g_multiClientManager.rtspClientData[clientId].monitorFlag = 1;
            if (scs.session) {
                g_multiClientManager.rtspClientData[i].clientHandle->sendTeardownCommand(
                        *scs.session, NULL);
            }

            // release the resource
            g_multiClientManager.activeClientCount--;
            g_multiClientManager.rtspClientData[i].urlData.erase(0);
            g_multiClientManager.rtspClientData[i].clientId = 0xFF;
            g_multiClientManager.rtspClientData[i].env = NULL;
            g_multiClientManager.rtspClientData[i].scheduler = NULL;
            g_multiClientManager.rtspClientData[i].clientHandle = NULL;
            g_multiClientManager.rtspClientData[i].clientState = CLIENT_STATUS_STOP;

            if (0 == g_multiClientManager.activeClientCount) {
                g_multiClientManager.initState = false;
                g_multiClientManager.callback = NULL;
                g_multiClientManager.threadPid = 0;
                if (!g_multiClientManager.rtspClientData.empty()) {
                    g_multiClientManager.rtspClientData.clear();
                }
                LOGD("clear all resource.\n");
            }
        }
        return RET_OK;
    }
    return RET_ERROR;
}

