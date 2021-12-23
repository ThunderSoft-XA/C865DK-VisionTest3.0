package com.thundercomm.rtsp;

import android.content.Context;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.media.MediaCodec;
import android.media.MediaFormat;
import android.text.TextUtils;
import android.util.Log;

import java.nio.ByteBuffer;
import java.util.concurrent.ArrayBlockingQueue;

/**
 * video decoder.
 *
 */
public class TsAudioDecoder {
    private static final String TAG = "TSVAudioDecoder";

    public MediaCodec mediaCodec;

    public Thread threadAudioDecoder = null;

    public ArrayBlockingQueue<TsPcmData> mPcmDatas;

    public boolean isDecoding = false;

    private AudioTrack audioTrack = null;

    private long timestamp2;

    private Context mContext;

    private MediaFormat format;

    private static final int NUM_8000 = 8000;
    private static final int NUM_10000 = 10000;
    private static final int NUM_30 = 30;
    private static final int NUM_6 = 6;
    private static final int NUM_100000 = 100000;

    /**
     * TsAudioDecoder construct.
     * @param queue video queue
     * @param context context
     */
    public TsAudioDecoder(ArrayBlockingQueue<TsPcmData> queue, Context context) {
        // TODO Auto-generated constructor stub
        this.mPcmDatas = queue;
        this.mContext = context;
    }

    public TseRtPlayState initDecoder() {
        init();
        return TseRtPlayState.RT_PLAY_STATE_SUCCESS;
    }

    /**
     * init audiotrack.
     *
     * @return int
     */
    private int init() {
        int minBufferSize = AudioTrack.getMinBufferSize(NUM_8000, AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT);
        audioTrack = new AudioTrack(AudioManager.STREAM_MUSIC, NUM_8000,
                AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT, minBufferSize * NUM_6,
                AudioTrack.MODE_STREAM);
        audioTrack.setVolume(1.0f);
        audioTrack.play();
        return 0;
    }

    /**
     * start decode.
     */
    public void start(String audioType) {
        init();
        try {
            // audioType(PCMA/PCMU),create mediaCodec;
            if (!TextUtils.isEmpty(audioType)) {
                if (audioType.equals("PCMA")) {
                    mediaCodec = MediaCodec
                            .createDecoderByType(MediaFormat.MIMETYPE_AUDIO_G711_ALAW);
                    format = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_G711_ALAW,
                            NUM_8000, 1);
                } else if (audioType.equals("PCMU")) {
                    mediaCodec = MediaCodec
                            .createDecoderByType(MediaFormat.MIMETYPE_AUDIO_G711_MLAW);
                    format = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_G711_MLAW,
                            NUM_8000, 1);
                } else {
                    Log.d(TAG, "not support audio type :" + audioType);
                    return;
                }
            }
            mediaCodec.configure(format, null, null, 0);
            if (mediaCodec == null) {
                return;
            }
            mediaCodec.start();
        } catch (Exception exception) {
            exception.printStackTrace();
            return;
        }
        if (!this.isDecoding) {
            this.isDecoding = true;
            this.decode();
            Log.i(TAG, "start end");
        }
    }

    // strat to decode
    private void decode() {
        Log.i(TAG, "decode start");
        // valued decoder and start mian thread

        this.threadAudioDecoder = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    doDecode();
                } catch (Exception exception) {
                    exception.printStackTrace();
                }
            }
        });
        threadAudioDecoder.setPriority(Thread.MIN_PRIORITY);
        threadAudioDecoder.start();
    }

    /**
     * stop decode.
     */
    public void stop() {
        this.isDecoding = false;
        this.threadAudioDecoder = null;
        // clear Queue
        if (null != mPcmDatas) {
            mPcmDatas.clear();
        }
    }

    /**
     * release decode.
     */
    public void release() {
        if (this.isDecoding) {
            this.stop();
        }
        // clear Queue
        if (null != mPcmDatas) {
            mPcmDatas.clear();
            mPcmDatas = null;
        }
        // stop and release decoder
        if (null != mediaCodec) {
            this.mediaCodec.stop();
            this.mediaCodec.release();
            this.mediaCodec = null;
        }
    }

    private void doDecode() {
        MediaCodec.BufferInfo bufferInfo = new MediaCodec.BufferInfo();
        while (isDecoding) {
            ByteBuffer[] inputBuffers = mediaCodec.getInputBuffers();
            ByteBuffer[] outputBuffers = mediaCodec.getOutputBuffers();
            int inputBufferIndex = mediaCodec.dequeueInputBuffer(NUM_10000);
            if (inputBufferIndex >= 0) {

                TsPcmData pcmDataPop = null;
                ByteBuffer inputBuffer = inputBuffers[inputBufferIndex];
                inputBuffer.clear();
                try {
                    if (null != mPcmDatas) {
                        pcmDataPop = mPcmDatas.take();
                    } else {
                        Log.e(TAG, "mPcmDatas is null!");
                    }
                } catch (InterruptedException exception) {
                    // TODO Auto-generated catch block
                    exception.printStackTrace();
                }
                if (pcmDataPop != null && pcmDataPop.getPcm() != null) {
                    inputBuffer.put(pcmDataPop.getPcm(), 0, pcmDataPop.getSize());
                }
                mediaCodec.queueInputBuffer(inputBufferIndex, 0, pcmDataPop.getSize(),
                        (timestamp2 * NUM_100000 / NUM_30), 0);
                // extractor.advance();
                timestamp2++;
            }
            int outputBufferIndex = mediaCodec.dequeueOutputBuffer(bufferInfo, 0);
            while (outputBufferIndex >= 0) {
                ByteBuffer buf = outputBuffers[outputBufferIndex];
                byte[] chunk = new byte[bufferInfo.size];
                buf.get(chunk);
                buf.clear();
                if (chunk.length > 0) {
                    audioTrack.write(chunk, 0, chunk.length);
                }
                mediaCodec.releaseOutputBuffer(outputBufferIndex, false);
                outputBufferIndex = mediaCodec.dequeueOutputBuffer(bufferInfo, 0);
            }
        }
    }
}
