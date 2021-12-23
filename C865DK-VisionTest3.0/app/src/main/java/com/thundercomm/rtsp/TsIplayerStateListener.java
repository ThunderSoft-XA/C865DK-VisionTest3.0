package com.thundercomm.rtsp;

/**
 * video play states listener.
 *
 *
 */
public interface TsIplayerStateListener {
    /**
     * play video.
     */
    void onPlay();

    /**
     * stop video.
     */
    void onStop();

    /**
     * Prepare Complete.
     */
    void onPrepareComplete();

    /**
     * error
     *
     * @param state TSERTPlayState
     */
    void onError(TseRtPlayState state);
}
