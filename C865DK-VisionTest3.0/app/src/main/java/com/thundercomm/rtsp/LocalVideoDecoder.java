package com.thundercomm.rtsp;

import android.graphics.ImageFormat;
import android.media.Image;
import android.media.ImageReader;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.util.Log;
import android.view.Surface;

import androidx.annotation.Nullable;

import com.thundercomm.eBox.Jni;
import com.thundercomm.eBox.Model.EventMessage;
import com.thundercomm.eBox.Utils.LogUtil;

import org.greenrobot.eventbus.EventBus;
import org.greenrobot.eventbus.Subscribe;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import lombok.SneakyThrows;


/**
 * video decoder.
 */
public class LocalVideoDecoder {

    private static final String TAG = "LocalVideoDecoder";
    private static final boolean VERBOSE = true;
    private static final boolean SAVEFILE = false;
    private boolean isPreview = false;
    private String path;


    public interface OnFrameAvailabkeListener {
        void onFrameAvailable(long timestamp, int index);
    }

    // decoder
    public MediaCodec mCodec;

    // format
    public MediaFormat mFormat;

    @Nullable
    public Surface mSurface = null;

    public Surface mInputSurface = null;

    public OnFrameAvailabkeListener mFrameListener = null;

    // decoder thread
    public Thread threadDecoder = null;

    // isDecoding
    public boolean isDecoding = false;

    public int clientid = 0;

    public int bufflength = 0;

    public static int GET_IMAGE_INTERVAL = 1_000 * 30;

    public ArrayBlockingQueue<TsPcmData> mPcmDatas = null;

    public TseDecodeType decodeType = null;

    private static final int NUM_10000 = 10000;
    private static final int DEFAULT_TIMEOUT = -1;

    private static final int NUM_5 = 5;

    private static final int NUM_7FA30C03 = 0x7FA30C03;

    private OnNetWorkErrorListener netWorkErrorListener = null;

    private TsPlayerDto playerDto;

    private final Lock mDecTimerLock = new ReentrantLock();
    private Condition mDecTimerCond = mDecTimerLock.newCondition();
    MediaExtractor mediaExtractor;
    MediaFormat mediaFormat;
    private static int TIMER_MEM_TASK_TIME = 1000 / 20;

    /**
     * LocalVideoDecoder construction.
     *
     * @param surface Surface
     * @param id      id
     */
    public LocalVideoDecoder(Surface surface, int id) {

        super();
        this.clientid = id;
        this.mSurface = surface;
        this.isPreview = true;
        this.init("video/avc");
        EventBus.getDefault().register(this);
        this.start();
    }

    /**
     * init decoder.
     *
     * @param type TSEDecodeType
     */
    TseRtPlayState initDecoder(TseDecodeType type, boolean isPreview) {
        this.isPreview = isPreview;
        this.decodeType = type;
        int res = -1;
        switch (this.decodeType) {
            case H264:
                res = this.init(playerDto.getDecoderTypeH264());
                break;
            case H265:
                res = this.init(playerDto.getDecoderTypeH265());
                break;
            default:
                break;
        }
        if (res < 0) {
            return TseRtPlayState.RT_PLAY_STATE_INIT_DEC_FAILURE;
        }
        return TseRtPlayState.RT_PLAY_STATE_SUCCESS;
    }

    private int init(String type) {
        try {
            this.mCodec = MediaCodec.createDecoderByType(type);
        } catch (IOException e) {
            e.printStackTrace();
        }
        mediaExtractor = new MediaExtractor();
        //设置MP4文件存放的位置
        int index = (clientid + 1);

        path = "/storage/emulated/0/EDGEBOX/ch2_" + index + ".mp4";
        try {
            mediaExtractor.setDataSource(path);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Log.d(TAG, "getTrackCount: " + mediaExtractor.getTrackCount());
        for (int i = 0; i < mediaExtractor.getTrackCount(); i++) {
            MediaFormat format = mediaExtractor.getTrackFormat(i);
            LogUtil.e(TAG, i + "编号通道格式" + format.getString(MediaFormat.KEY_MIME));
            String mime = format.getString(MediaFormat.KEY_MIME);
//            mediaFormat = format;
//            mediaExtractor.selectTrack(0);
            //如果是video格式
            if (mime.startsWith("video")) {
                mediaFormat = format;
                mediaExtractor.selectTrack(0);
            }
        }
        // configure
        if (true) {
            this.mCodec.configure(mediaFormat, mSurface, null, 0);
        } else {
            this.mCodec.configure(mFormat, null, null, 0);
        }
        // start decoder
        this.mCodec.start();
        return 0;
    }

    /**
     * start decode.
     */
    public void start() {
        Log.i(TAG, "start");
        if (!this.isDecoding) {
            this.isDecoding = true;
            this.decode();
            Log.i(TAG, "start end");
        }
    }

    /**
     * stop decode.
     */
    public void stop() {
        this.isDecoding = false;
        // clear Queue
        if (null != mPcmDatas) {
            mPcmDatas.clear();
        }
        if (null != this.threadDecoder) {
            this.threadDecoder.interrupt();
            try {
                this.threadDecoder.join();
            } catch (InterruptedException exception) {
                // TODO Auto-generated catch block
                exception.printStackTrace();
            }
            this.threadDecoder = null;
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
        if (null != mCodec) {
            this.mCodec.stop();
            this.mCodec.release();
            this.mCodec = null;
        }
    }

    long pts = System.currentTimeMillis();
    int frame = 0;

    // strat to decode
    private void decode() {
        Jni.Affinity.bindToCpu(clientid % 4 + 4);
        Log.i(TAG, "decode start");
        // valued decoder and start mian thread
        this.threadDecoder = new Thread(new Runnable() {
            @SneakyThrows
            @Override
            public void run() {
                long next_dec_time = System.currentTimeMillis();
                ;
                long dnext_dec_time = next_dec_time;
//                doDecode();
                while (true) {
                    int inputIndex = mCodec.dequeueInputBuffer(-1);
                    Log.d(TAG, "inputIndex: " + inputIndex);
                    if (inputIndex >= 0) {
                        ByteBuffer byteBuffer = mCodec.getInputBuffer(inputIndex);
                        //读取一片或者一帧数据
                        int sampSize = mediaExtractor.readSampleData(byteBuffer, 0);
                        //读取时间戳
                        long time = mediaExtractor.getSampleTime();
                        Log.d(TAG, "sampSize: " + sampSize + "time: " + time);
                        if (sampSize > 0) {
                            mCodec.queueInputBuffer(inputIndex, 0, sampSize, time, 0);
                            //读取一帧后必须调用，提取下一帧
                            mediaExtractor.advance();
                            mDecTimerLock.lock();
                            try {
                                dnext_dec_time = System.currentTimeMillis() - next_dec_time;
                                if (dnext_dec_time < TIMER_MEM_TASK_TIME) {
                                    dnext_dec_time = TIMER_MEM_TASK_TIME - dnext_dec_time;
                                    mDecTimerCond.awaitNanos(dnext_dec_time * 1000 * 1000);
                                }
                            } finally {
                                mDecTimerLock.unlock();
                            }
                            next_dec_time = System.currentTimeMillis();
                        } else {
                            try {
                                mediaExtractor = null;
                                mediaExtractor = new MediaExtractor();
                                mediaExtractor.setDataSource(path);
                            } catch (IOException e) {
                                e.printStackTrace();
                            }
                            Log.d(TAG, "getTrackCount: " + mediaExtractor.getTrackCount());
                            for (int i = 0; i < mediaExtractor.getTrackCount(); i++) {
                                MediaFormat format = mediaExtractor.getTrackFormat(i);
                                LogUtil.e(TAG, i + "编号通道格式" + format.getString(MediaFormat.KEY_MIME));
                                String mime = format.getString(MediaFormat.KEY_MIME);
                                //如果是video格式
                                if (mime.startsWith("video")) {
                                    mediaFormat = format;
                                    mediaExtractor.selectTrack(0);
                                }
                            }
                        }

                    }
                    MediaCodec.BufferInfo bufferInfo = new MediaCodec.BufferInfo();
                    int outIndex = mCodec.dequeueOutputBuffer(bufferInfo, 0);
                    Log.d(TAG, "outIndex: " + outIndex);
                    if (outIndex >= 0) {
                        mCodec.releaseOutputBuffer(outIndex, true);
                        frame++;
                    }
                    if (System.currentTimeMillis() - pts >= 1000 * 10) {
                        LogUtil.d(TAG, "id:" + clientid + " --->" + "Decode Speed ===>" + frame / 10 + "fps");
                        frame = 0;
                        pts = System.currentTimeMillis();
                    }
                }

            }
        });
        threadDecoder.start();
    }


    /**
     * main process of decode.
     */
    private void doDecode() {
        try {
            long next_dec_time = System.currentTimeMillis();
            long dnext_dec_time = next_dec_time;
            TIMER_MEM_TASK_TIME = 1000 / ((playerDto.getVideoFrameRate() + 9) / 10 * 10);
            ByteBuffer dstBuf;
            ByteBuffer[] inputBuffers;
            Log.i(TAG, "doDecode start");
            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            inputBuffers = mCodec.getInputBuffers();
            while (isDecoding) {
                if (threadDecoder.isInterrupted()) {
                    Log.i(TAG, "isInterrupted");
                    break;
                }
                // Queue Input buffer
                mDecTimerLock.lock();
                try {
                    dnext_dec_time = System.currentTimeMillis() - next_dec_time;
                    if (dnext_dec_time < TIMER_MEM_TASK_TIME) {
                        dnext_dec_time = TIMER_MEM_TASK_TIME - dnext_dec_time;
                        mDecTimerCond.awaitNanos(dnext_dec_time * 1000 * 1000);

                    }
                } finally {
                    mDecTimerLock.unlock();
                }
                next_dec_time = System.currentTimeMillis();
                int inputBufferIndex = mCodec.dequeueInputBuffer(DEFAULT_TIMEOUT);
                if (inputBufferIndex >= 0) {
                    dstBuf = inputBuffers[inputBufferIndex];
                    dstBuf.clear();
                    TsPcmData pcmDataPop = null;
                    try {
                        if (null != mPcmDatas) {
                            if (mPcmDatas.size() >= 31) {
                                while (mPcmDatas.size() > 0) {
                                    pcmDataPop = mPcmDatas.take();
                                    int IDRValue = 5;
                                    if (decodeType == TseDecodeType.H264) {
                                        IDRValue = 5;
                                        if (pcmDataPop.nalu_type == IDRValue) {
                                            LogUtil.e(TAG, "Drop Frame H264 - util to IDR size " + mPcmDatas.size());
                                            break;
                                        }
                                    } else {
                                        //LogUtil.d(TAG, "H265 decode not drop");
                                        IDRValue = 19;
                                        if (pcmDataPop.nalu_type >= IDRValue
                                                && pcmDataPop.nalu_type <= (IDRValue + 1)) {
                                            LogUtil.e(TAG, "Drop Frame H265 - util to IDR size " + mPcmDatas.size());
                                            break;
                                        }
                                    }
                                }
                            } else {
                                pcmDataPop = mPcmDatas.take();
                            }
                            //LogUtil.e(TAG, "NAUL " + pcmDataPop.getNalutype(decodeType));
                        } else {
                            Log.e(TAG, "mPcmDatas is null!");
                        }
                    } catch (InterruptedException exception) {
                        // TODO Auto-generated catch block
                        exception.printStackTrace();
                    }
                    int sampleSize = -1;
                    if (pcmDataPop != null && pcmDataPop.getPcm() != null) {
                        sampleSize = pcmDataPop.getSize();
                    }

                    if (sampleSize >= 0) {
                        bufflength = pcmDataPop.getSize();
                        if (pcmDataPop.getSize() == NUM_5
                                && new String(pcmDataPop.getPcm()).equals("error")) {
                            Log.d("xuecq0313", "error");
                            if (netWorkErrorListener != null) {
                                netWorkErrorListener.onNetWorkError();
                                mSurface.release();
                                mCodec.stop();
                                mCodec.flush();
                                mCodec.release();
                            }
                            break;
                        }
                        dstBuf.put(pcmDataPop.getPcm(), 0, bufflength);
                    }
                    if (sampleSize > 0) {
                        mCodec.queueInputBuffer(inputBufferIndex, 0, bufflength, 0, 0);
                    }
                }
                // Queue Output buffer
                int outputBufferIndex = mCodec.dequeueOutputBuffer(info, NUM_10000);
                if (outputBufferIndex >= 0) {
                    if (mFrameListener != null) {
                        mFrameListener.onFrameAvailable(info.presentationTimeUs, outputBufferIndex);
                    }
                } else if (outputBufferIndex == MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED) {
                    // No need to update output buffer, since we don't touch it
                } else if (outputBufferIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    final MediaFormat oformat = mCodec.getOutputFormat();
                    getColorFormat(oformat);
                }
            }
        } catch (Exception exception) {
            exception.printStackTrace();
            Log.i(TAG, "decode start Exception");
        }
    }

    private boolean notKeyFrame(byte[] pcm) {
        int nalu = 0;
        switch (decodeType) {
            case H264:
                if (pcm.length >= 32) {
                    nalu = pcm[32] & 0x1f;
                }
                break;
            case H265:
                if (pcm.length >= 32) {
                    nalu = (pcm[32] & 0x7E) >> 1;
                }
                break;
            default:
                break;
        }
        LogUtil.e(TAG, "NAUL -----" + nalu);
        if (nalu == 1) {
            return true;
        } else {
            return false;
        }

    }

    private void sendImgToAlg(Image image) {

        LogUtil.e("Send Image " + image);
    }

    public interface OnNetWorkErrorListener {
        void onNetWorkError();
    }

    public void setNetWorkErrorListener(OnNetWorkErrorListener listener) {
        this.netWorkErrorListener = listener;
    }

    /**
     * release OutputBuffer.
     *
     * @param index index
     */
    public void render(int index) {
        this.mCodec.releaseOutputBuffer(index, true);
    }

    /**
     * get decode yuv format.
     *
     * @param format MediaFormat
     * @return decode yuv format
     */
    public String getColorFormat(MediaFormat format) {
        int colorFormat = format.getInteger(MediaFormat.KEY_COLOR_FORMAT);

        int color = NUM_7FA30C03;

        String formatString = "";
        if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_Format12bitRGB444) {
            formatString = "COLOR_Format12bitRGB444";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_Format16bitARGB1555) {
            formatString = "COLOR_Format16bitARGB1555";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_Format16bitARGB4444) {
            formatString = "COLOR_Format16bitARGB4444";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_Format16bitBGR565) {
            formatString = "COLOR_Format16bitBGR565";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_Format16bitRGB565) {
            formatString = "COLOR_Format16bitRGB565";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_Format18bitARGB1665) {
            formatString = "COLOR_Format18bitARGB1665";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_Format18BitBGR666) {
            formatString = "COLOR_Format18BitBGR666";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_Format18bitRGB666) {
            formatString = "COLOR_Format18bitRGB666";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_Format19bitARGB1666) {
            formatString = "COLOR_Format19bitARGB1666";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_Format24BitABGR6666) {
            formatString = "COLOR_Format24BitABGR6666";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_Format24bitARGB1887) {
            formatString = "COLOR_Format24bitARGB1887";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_Format24BitARGB6666) {
            formatString = "COLOR_Format24BitARGB6666";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_Format24bitBGR888) {
            formatString = "COLOR_Format24bitBGR888";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_Format24bitRGB888) {
            formatString = "COLOR_Format24bitRGB888";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_Format25bitARGB1888) {
            formatString = "COLOR_Format25bitARGB1888";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_Format32bitARGB8888) {
            formatString = "COLOR_Format32bitARGB8888";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_Format32bitBGRA8888) {
            formatString = "COLOR_Format32bitBGRA8888";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_Format8bitRGB332) {
            formatString = "COLOR_Format8bitRGB332";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatCbYCrY) {
            formatString = "COLOR_FormatCbYCrY";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatCrYCbY) {
            formatString = "COLOR_FormatCrYCbY";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatL16) {
            formatString = "COLOR_FormatL16";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatL2) {
            formatString = "COLOR_FormatL2";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatL24) {
            formatString = "COLOR_FormatL24";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatL32) {
            formatString = "COLOR_FormatL32";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatL4) {
            formatString = "COLOR_FormatL4";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatL8) {
            formatString = "COLOR_FormatL8";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatMonochrome) {
            formatString = "COLOR_FormatMonochrome";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatRawBayer10bit) {
            formatString = "COLOR_FormatRawBayer10bit";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatRawBayer8bit) {
            formatString = "COLOR_FormatRawBayer8bit";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities
                .COLOR_FormatRawBayer8bitcompressed) {
            formatString = "COLOR_FormatRawBayer8bitcompressed";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYCbYCr) {
            formatString = "COLOR_FormatYCbYCr";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYCrYCb) {
            formatString = "COLOR_FormatYCrYCb";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV411PackedPlanar) {
            formatString = "COLOR_FormatYUV411PackedPlanar";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV411Planar) {
            formatString = "COLOR_FormatYUV411Planar";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420PackedPlanar) {
            formatString = "COLOR_FormatYUV420PackedPlanar";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities
                .COLOR_FormatYUV420PackedSemiPlanar) {
            formatString = "COLOR_FormatYUV420PackedSemiPlanar";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV422PackedPlanar) {
            formatString = "COLOR_FormatYUV422PackedPlanar";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities
                .COLOR_FormatYUV422PackedSemiPlanar) {
            formatString = "COLOR_FormatYUV422PackedSemiPlanar";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV422Planar) {
            formatString = "COLOR_FormatYUV422Planar";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities
                .COLOR_FormatYUV422PackedSemiPlanar) {
            formatString = "COLOR_FormatYUV422PackedSemiPlanar";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV422Planar) {
            formatString = "COLOR_FormatYUV422Planar";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV422SemiPlanar) {
            formatString = "COLOR_FormatYUV422SemiPlanar";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV444Interleaved) {
            formatString = "COLOR_FormatYUV444Interleaved";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities
                .COLOR_QCOM_FormatYUV420SemiPlanar) {
            formatString = "COLOR_QCOM_FormatYUV420SemiPlanar";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities
                .COLOR_TI_FormatYUV420PackedSemiPlanar) {
            formatString = "COLOR_TI_FormatYUV420PackedSemiPlanar";
        } else if (colorFormat == color) {
            formatString = "QOMX_COLOR_FormatYUV420PackedSemiPlanar64x32Tile2m8ka";
        } else if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar) {
            formatString = "COLOR_FormatYUV420Planar";
        }

        Log.i("TAG", formatString);
        return formatString;
    }

    private static String getImageFormat(Image image) {
        int format = image.getFormat();
        switch (format) {
            case ImageFormat.YUV_420_888:
                return "YUV_420_888";
            case ImageFormat.NV21:
                return "NV21";
            case ImageFormat.YV12:
                return "YV12";
            default:
                return "";
        }
    }

    private static boolean isImageFormatSupported(Image image) {
        int format = image.getFormat();
        switch (format) {
            case ImageFormat.YUV_420_888:
            case ImageFormat.NV21:
            case ImageFormat.YV12:
                return true;
        }
        return false;
    }

    public static byte[] subBytes(byte[] src, int begin, int count) {
        byte[] bs = new byte[count];
        System.arraycopy(src, begin, bs, 0, count);
        return bs;
    }


    public ImageReader.OnImageAvailableListener imageAvailableListener = new ImageReader.OnImageAvailableListener() {
        @Override
        public void onImageAvailable(ImageReader reader) {
            Image image = reader.acquireNextImage();
            if (image != null) {
                image.close();
            }
        }
    };

    private static void dumpFile(String fileName, byte[] data) {
        FileOutputStream outStream;
        try {
            outStream = new FileOutputStream(fileName);
        } catch (IOException ioe) {
            throw new RuntimeException("Unable to create output file " + fileName, ioe);
        }
        try {
            outStream.write(data);
            outStream.close();
        } catch (IOException ioe) {
            throw new RuntimeException("failed writing data to file " + fileName, ioe);
        }
    }

    @Subscribe
    public void getEventMsg(EventMessage eventMessage) {
        if (eventMessage.getMessageType().equals("LOST_NETWORK")) {
            LogUtil.e(TAG, "LOST_NETWORK");
            if (netWorkErrorListener != null) {
                netWorkErrorListener.onNetWorkError();
                mSurface.release();
                mCodec.stop();
                mCodec.flush();
                mCodec.release();
            }
        }
    }

    private boolean putBufferToCoder(MediaExtractor extractor, MediaCodec decoder, ByteBuffer[] inputBuffers) {
        boolean isMediaEOS = false;
        int inputBufferIndex = decoder.dequeueInputBuffer(1000);
        if (inputBufferIndex >= 0) {
            ByteBuffer inputBuffer = inputBuffers[inputBufferIndex];
            int sampleSize = extractor.readSampleData(inputBuffer, 0);
            if (sampleSize < 0) {
                decoder.queueInputBuffer(inputBufferIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                isMediaEOS = true;
                Log.v(TAG, "media eos");
            } else {
                decoder.queueInputBuffer(inputBufferIndex, 0, sampleSize, extractor.getSampleTime(), 0);
                extractor.advance();
            }
        }
        return isMediaEOS;
    }
}
