package com.thundercomm.rtsp;

import android.content.Context;
import android.text.TextUtils;
import android.util.Log;
import android.view.Surface;

import com.thundercomm.eBox.Model.EventMessage;
import com.thundercomm.eBox.Utils.LogUtil;

import org.greenrobot.eventbus.EventBus;

import java.util.concurrent.ArrayBlockingQueue;

/**
 * real time player.
 */
public class TsRealTimePlayer {

    private static final String TAG = "TSRealTimePlayer";

    private static volatile TsRealTimePlayer mRealTimePlayer = null;

    private Surface mSurface = null;

    private TseDecodeType decodeType = null;

    private TsVideoDecoder videoDecoder = null;

    private TsAudioDecoder audioDecoder = null;

    private TsRtspNative mTsrtpNative = null;

    private TsIplayerStateListener mListener = null;

    private Context mContext = null;

    private static final int PLAYER_ONPLAY = 0;

    private static final int PLAYER_ONSTOP = 1;

    private static final int PLAYER_ONPREPARECOMPLETE = 2;

    private static final int PLAYER_ONERROR = 3;

    private boolean isPlaying = false;

    private boolean isPrepared = false;

    private String audioType;

    private static final int NUM_3 = 3;

    private TsPlayerDto playerDto;

    private boolean isOnPreview;

    private int clientid;

    private int frame = 0;
    private long pts = 0;

    /**
     * TsRealTimePlayer get instance.
     *
     * @param context context
     * @return TsRealTimePlayer
     */
    public static TsRealTimePlayer getInstance(Context context) {
        if (mRealTimePlayer == null) {
            synchronized (TsRealTimePlayer.class) {
                if (mRealTimePlayer == null) {
                    mRealTimePlayer = new TsRealTimePlayer(context);
                }
            }
        }
        return mRealTimePlayer;
    }

    /**
     * TsRealTimePlayer Constructor.
     *
     * @param context context
     */
    public TsRealTimePlayer(Context context) {
        super();
        mContext = context;
        playerDto = new TsPlayerDto();
    }

    /**
     * set decode type.
     *
     * @param decodeType TSEDecodeType
     */
    public void setDecodeType(TseDecodeType decodeType) {
        this.decodeType = decodeType;
    }

    /**
     * set surface.
     *
     * @param surface Surface
     */
    public void setSurface(Surface surface) {
        this.mSurface = surface;
    }

    /**
     * set player listener.
     *
     * @param listener TSIPlayerStateListener
     */
    public void setPlayerListener(TsIplayerStateListener listener) {
        this.mListener = listener;
    }

    /**
     * prepare for realtime player.
     */
    public void prepare(int id, boolean isOnPreview) {
        clientid = id;
        if (this.isPlaying) {
            return;
        }
        this.isOnPreview = isOnPreview;
        if (null == this.mTsrtpNative) {
            getTsRtspNative(playerDto.getPcmQueueForVideo(), id);
        }

        if (null == this.videoDecoder) {
            if (null == this.mSurface) {
                this.performPlayListener(PLAYER_ONERROR, TseRtPlayState.RT_PLAY_STATE_NO_SURFACE);
                Log.i(TAG, "prepare: surface is null");
                return;
            }
            netWorkErrorListener(playerDto.getPcmQueueForVideo(), id);

        }

        if (null == this.decodeType) {
            this.decodeType = playerDto.getDefaultType();
        }

        TseRtPlayState state = this.videoDecoder.initDecoder(this.decodeType, isOnPreview);
        LogUtil.e(TAG, "initDecoder");

        if (state != TseRtPlayState.RT_PLAY_STATE_SUCCESS) {
            this.performPlayListener(PLAYER_ONERROR, state);
        }

        // TODO TEST CODE
        // TSCameraController.getInstance().getSPSAndPPS(this.spsAndPPSCallBack);
        this.isPrepared = true;
        this.performPlayListener(PLAYER_ONPREPARECOMPLETE, null);
        Log.i(TAG, "prepare end");
    }

    private void getTsRtspNative(ArrayBlockingQueue<TsPcmData> queue, final int id) {
        this.mTsrtpNative = new TsRtspNative(playerDto.getPcmQueueForVideo(), id);
    }

    private long currTime = 0;

    private void netWorkErrorListener(ArrayBlockingQueue<TsPcmData> queue, final int id) {
        this.videoDecoder = new TsVideoDecoder(this.mSurface, queue,
                new TsVideoDecoder.OnFrameAvailabkeListener() {
                    @Override
                    public void onFrameAvailable(long timestamp, int index) {
                        videoDecoder.render(index);
                        frame++;
                        if (System.currentTimeMillis() - currTime >= 1000) {
                            LogUtil.i(TAG, "Rtsp id:" + id + " decode speed: " + frame + "/s");
                            EventBus.getDefault().post(new EventMessage("FRAME", id, frame));
                            frame = 0;
                            currTime = System.currentTimeMillis();
                        }
                    }
                });
        videoDecoder.setNetWorkErrorListener(new TsVideoDecoder.OnNetWorkErrorListener() {
            @Override
            public void onNetWorkError() {
                mTsrtpNative.rtspClientCloseJni(id);
                performPlayListener(PLAYER_ONERROR, TseRtPlayState.RT_PLAY_STATE_NETWORK_ERROR);
            }
        });
    }

    /**
     * start realtime player.
     */
    public void start(String url, int id, boolean isOnPreview) {
        clientid = id;
        if (this.isPrepared && !this.isPlaying) {
            this.mTsrtpNative.start(url, id);
            this.videoDecoder.clientid = id;
            this.videoDecoder.start();
            this.isPlaying = true;
            this.performPlayListener(PLAYER_ONPLAY, null);
            Log.i(TAG, "start");
        }

    }

    /**
     * close streaming.
     *
     * @param id ipc client id
     */
    public void closeRtspClient(int id) {
        if (mTsrtpNative != null) {
            mTsrtpNative.rtspClientCloseJni(id);
        }
    }

    /**
     * get video encode type.
     *
     * @return string
     */
    public String getVideoType() {
        if (mTsrtpNative == null) {
            return null;
        }
        return mTsrtpNative.getVideoType();
    }

    /**
     * start audio thread.
     *
     * @param id ipc client id
     */
    public void audioStart(int id) {
        audioType = mTsrtpNative.getAudioType();
        if (TextUtils.isEmpty(audioType)) {
            return;
        }
        if (null == this.audioDecoder) {
            this.audioDecoder = new TsAudioDecoder(TsPlayerDto.getPcmQueueForAudio(),
                    mContext);

        }
        this.audioDecoder.start(audioType);
        Log.i(TAG, "audioStart");
    }

    /**
     * stop realtime player.
     */
    public void stop(int id) {
        if (this.isPlaying) {
            this.mTsrtpNative.stop(id);
            this.isPlaying = false;
            this.isPrepared = false;
            this.videoDecoder.stop();
            // this.audioDecoder.stop();
            this.performPlayListener(PLAYER_ONSTOP, null);
            Log.i(TAG, "stop");
        }
    }

    // stop/start audio set flag.
    public void setIsStartAudio(boolean isStartAudio) {
        mTsrtpNative.setStartAudio(isStartAudio);
    }

    /**
     * stop audio thread.
     */
    public void stopAudio() {
        if (audioDecoder != null) {
            this.audioDecoder.stop();
            audioDecoder = null;
        }
    }

    /**
     * release realtime player.
     */
    public void release(int id) {
        if (this.isPlaying) {
            this.stop(id);
        }
        if (null != this.videoDecoder) {
            this.videoDecoder.release();
            this.videoDecoder = null;
        }
        if (null != this.audioDecoder) {
            this.audioDecoder.release();
            this.audioDecoder = null;
        }
        this.mTsrtpNative = null;
        this.isPlaying = false;
        this.isPrepared = false;
        this.decodeType = null;
        this.mSurface = null;
        this.mListener = null;
        mRealTimePlayer = null;
        Log.i(TAG, "release");
    }

    /**
     * realtime player is playing or not.
     *
     * @return true/false
     */
    public boolean isPlaying() {
        return this.isPlaying;

    }

    private void performPlayListener(int playFunc, TseRtPlayState state) {
        if (mListener != null) {
            switch (playFunc) {
                case PLAYER_ONPLAY:
                    this.mListener.onPlay();
                    break;
                case PLAYER_ONPREPARECOMPLETE:
                    this.mListener.onPrepareComplete();
                    break;
                case PLAYER_ONSTOP:
                    this.mListener.onStop();
                    break;
                case PLAYER_ONERROR:
                    this.mListener.onError(state);
                    break;
                default:
                    break;
            }
        }
    }
}
