package com.thundercomm.rtsp;

import java.util.concurrent.ArrayBlockingQueue;

/**
 * Play data transfer object.
 */
public class TsPlayerDto {
    // Array blocking queue used for receiving data from RTP server
    private ArrayBlockingQueue<TsPcmData> videoQueue
            = new ArrayBlockingQueue<TsPcmData>(
            NUM_1000);

    private static final ArrayBlockingQueue<TsPcmData> audioQueue
            = new ArrayBlockingQueue<TsPcmData>(
            100);
    // Data port number
    private static final String DECODER_TYPE_H264 = "video/avc";// android media codec type h264

    private static final String DECODER_TYPE_H265 = "video/hevc";

    private static final int VIDEO_W = 1080;// resolution width

    private static final int VIDEO_H = 1920;

    private static final int NUM_100 = 100;

    private static final int NUM_1000 = 1000;

    private static final int VIDEO_FRAME_RATE = 25;//

    private static final int VIDEO_BIT_RATE = (int) (1024 * 0.1);

    private static final TseDecodeType DEFAULT_TYPE = TseDecodeType.H264;

    private static TsPlayerDto mTsPlayerDTO = null;

    /**
     * get TSPlayerDTO instance.
     */
    public TsPlayerDto() {
        super();
    }

    /**
     * get Array blocking queue used for receiving data from RTP server.
     *
     * @return Array blocking queue used for receiving data from RTP server
     */
    public ArrayBlockingQueue<TsPcmData> getPcmQueueForVideo() {
        return videoQueue;
    }

    public static ArrayBlockingQueue<TsPcmData> getPcmQueueForAudio() {
        return audioQueue;
    }

    /**
     * get android media codec type(H264).
     *
     * @return android media codec type(H264)
     */
    public String getDecoderTypeH264() {
        return DECODER_TYPE_H264;
    }

    /**
     * get android media codec type(H265).
     *
     * @return android media codec type(H265)
     */
    public String getDecoderTypeH265() {
        return DECODER_TYPE_H265;
    }

    /**
     * width resolution.
     *
     * @return width resolution
     */
    public int getVideoW() {
        return VIDEO_W;
    }

    /**
     * height resolution.
     *
     * @return height resolution
     */
    public int getVideoH() {
        return VIDEO_H;
    }

    /**
     * get video frame rate.
     *
     * @return video frame rate
     */
    public int getVideoFrameRate() {
        return VIDEO_FRAME_RATE;
    }

    /**
     * get video bit rate.
     *
     * @return video bit rate
     */
    public int getVideoBitRate() {
        return VIDEO_BIT_RATE;
    }

    /**
     * get default decode type.
     *
     * @return default decode type
     */
    public TseDecodeType getDefaultType() {
        return DEFAULT_TYPE;
    }

}
