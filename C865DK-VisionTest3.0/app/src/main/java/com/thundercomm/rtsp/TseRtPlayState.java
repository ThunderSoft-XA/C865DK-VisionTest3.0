package com.thundercomm.rtsp;

/**
 * RTP play state enum.
 * 
 */
public enum TseRtPlayState {
    /**
     * RTP play state : NoINPUT.
     */
    RT_PLAY_STATE_NOINPUT,
    /**
     * RTP play state : ERROR.
     */
    RT_PLAY_STATE_ERROR,
    /**
     * RTP play state : PLAYING.
     */
    RT_PLAY_STATE_PLAYING,
    /**
     * RTP play state : STOP.
     */
    RT_PLAY_STATE_STOP,
    /**
     * RTP play state : PARESYNC.
     */
    RT_PLAY_STATE_PARESYNC,
    /**
     * RTP play state : PARECOMPLETE.
     */
    RT_PLAY_STATE_PARECOMPLETE,
    /**
     * RTP play state : NO SURFACE.
     */
    RT_PLAY_STATE_NO_SURFACE,
    /**
     * RTP play state : PREPARE ERROR.
     */
    RT_PLAY_STATE_SUCCESS,
    /**
     * RTP play state : PREPARE ERROR.
     */
    RT_PLAY_STATE_PREPARE_ERROR,
    /**
     * RTP play state : INIT DEC FAILURE.
     */
    RT_PLAY_STATE_INIT_DEC_FAILURE,
    /**
     * RTP play state : GET PPSSPS FAILURE.
     */
    RT_PLAY_STATE_GET_PPSSPS_FAILURE,
    /**
     * RTP play state : GET PPSSPS OVERTIME.
     */
    RT_PLAY_STATE_GET_PPSSPS_OVERTIME,

    RT_PLAY_STATE_NETWORK_ERROR,

}
