package com.thundercomm.rtsp;

import android.text.TextUtils;
import android.util.Log;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.ArrayBlockingQueue;

/**
 * RTP native.

 */
public class TsRtspNative {
    private static boolean isInitiated = false;

    private ArrayBlockingQueue<TsPcmData> mVideoPcmDatas = null;

    private boolean startAudio = false;

    private int clientid = 0;

    public String audioType = null;

    public String videoType = null;

    private static final int NUM_3 = 3;

    static {
        System.loadLibrary("RtspClientJni");
    }

    /**
     * native method : close rtsp stream.
     *
     * @param clientId current client id
     * @return int
     */
    public native int rtspClientCloseJni(int clientId);

    /**
     * native method: open rtsp.
     *
     * @param etUrl    rtsp url
     * @param clientId current client id([0,3])
     * @return int
     */
    public native int rtspClientOpenJni(String etUrl, int clientId);

    /**
     * TSRTPNative construction.
     *
     * @param videoDatas video stream packet queue
     * @param id         current client id [0，3]
     */
    public TsRtspNative(ArrayBlockingQueue<TsPcmData> videoDatas, int id) {
        mVideoPcmDatas = videoDatas;
        mVideoPcmDatas.clear();
        isInitiated = true;
    }

    /**
     * RTP stop video stream packet receiving thread.
     */
    public void stop(int clientId) {
        Log.e("native", "java call stop");
        rtspClientCloseJni(clientId);
    }

    private TsPcmData putVideoData(int what, byte[] callbackBuffer) {
        if (!TextUtils.isEmpty(videoType) && videoType.equals("Err")) {
            String error = "error";
            byte[] data = new byte[error.length()];
            System.arraycopy(error.getBytes(), 0, data, 0, error.length());
            return new TsPcmData(data, error.length());
        } else {
            byte[] data = new byte[what];
            System.arraycopy(callbackBuffer, 0, data, 0, what);
            //debugsaveStreamFile(data, GlobalConfig.SAVE_PATH, "stream_"/*+System.currentTimeMillis()*/+"_rtsp.h264");
            return new TsPcmData(data, what);
        }
    }

    public static void debugsaveStreamFile(byte[] bfile, String filePath, String fileName) {
        BufferedOutputStream bos = null;
        FileOutputStream fos = null;

        File file = null;
        try {
            //通过创建对应路径的下是否有相应的文件夹。
            File dir = new File(filePath);
            if (!dir.exists()) {// 判断文件目录是否存在
                //如果文件存在则删除已存在的文件夹。
                dir.mkdirs();
            }

            //如果文件存在则删除文件
            file = new File(filePath, fileName);
            if(file.exists()){
                fos = new FileOutputStream(file, true);
            } else {
                fos = new FileOutputStream(file);
            }
            bos = new BufferedOutputStream(fos);
            //把需要保存的文件保存到SD卡中
            bos.write(bfile);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (bos != null) {
                try {
                    bos.close();
                } catch (IOException e1) {
                    e1.printStackTrace();
                }
            }
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException e1) {
                    e1.printStackTrace();
                }
            }
        }
    }

    public static void debugsaveperFrameFile(byte[] bfile, String filePath, String fileName) {
        BufferedOutputStream bos = null;
        FileOutputStream fos = null;

        File file = null;
        try {
            //通过创建对应路径的下是否有相应的文件夹。
            File dir = new File(filePath);
            if (!dir.exists()) {// 判断文件目录是否存在
                //如果文件存在则删除已存在的文件夹。
                dir.mkdirs();
            }

            //如果文件存在则删除文件
            file = new File(filePath, fileName);
            if(file.exists()){
                file.delete();
            }
            fos = new FileOutputStream(file);
            bos = new BufferedOutputStream(fos);
            //把需要保存的文件保存到SD卡中
            bos.write(bfile);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (bos != null) {
                try {
                    bos.close();
                } catch (IOException e1) {
                    e1.printStackTrace();
                }
            }
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException e1) {
                    e1.printStackTrace();
                }
            }
        }
    }

    /**
     * jni call java video callback.
     *
     * @param clientId           video id
     * @param what               what
     * @param callbackBuffer     callback buffer
     * @param presentationTimeUs time
     * @param videoType          Err (Network disconnection)
     */
    public void postDataFromNative(int clientId, int nalu_type, final int what, final byte[] callbackBuffer,
                                   long presentationTimeUs, String videoType) {
        this.videoType = videoType;
        if (isInitiated) {
            try {
                TsPcmData pcmData = putVideoData(what, callbackBuffer);
                pcmData.nalu_type = nalu_type;
                mVideoPcmDatas.put(pcmData);
            } catch (InterruptedException exception) {
                exception.printStackTrace();
            }
        }
    }

    int pos = -1;

    /**
     * jni call java audio callback.
     *
     * @param clientId           audio id
     * @param what               what
     * @param callbackBuffer     callback buffer
     * @param audioType          PCMA/PCMU
     * @param presentationTimeUs time
     */
    private void postAudioDataFromNative(int clientId, int what, byte[] callbackBuffer,
                                         String audioType, long presentationTimeUs) {
        this.audioType = audioType;
        if (!startAudio) {
            pos = -1;
            return;
        }
        Log.d("xuecq0324", "audio start");
        byte[] data;
        data = Arrays.copyOfRange(callbackBuffer, 0, callbackBuffer.length);
        TsPcmData pcmData = new TsPcmData(data, what);
        try {
            TsPlayerDto.getPcmQueueForAudio().put(pcmData);
        } catch (InterruptedException exception) {
            exception.printStackTrace();
        }
    }

    public void setStartAudio(boolean isStart) {
        startAudio = isStart;
    }

    public String getAudioType() {
        return audioType;
    }

    public String getVideoType() {
        return videoType;
    }

    /**
     * play video .
     *
     * @param url rtsp url
     * @param id  client id
     */
    public void start(final String url, final int id) {
        clientid = id;
        synchronized (this) {
            if (url != null && !url.equals("")) {
                rtspClientOpenJni(url, id);
            }
        }
    }

}
