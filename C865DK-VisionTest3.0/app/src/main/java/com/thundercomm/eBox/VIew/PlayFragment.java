package com.thundercomm.eBox.VIew;

import android.annotation.SuppressLint;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.Point;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.Layout;
import android.text.StaticLayout;
import android.text.TextPaint;
import android.text.TextUtils;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentTransaction;
import com.thundercomm.eBox.Config.GlobalConfig;
import com.thundercomm.eBox.EventBusManager;
import com.thundercomm.eBox.Gl.MyGLSurfaceView;
import com.thundercomm.eBox.Jni;
import com.thundercomm.eBox.Model.EventMessage;
import com.thundercomm.eBox.Model.RtspItemCollection;
import com.thundercomm.eBox.R;
import com.thundercomm.eBox.Utils.LogUtil;
import com.thundercomm.gateway.data.DeviceCollection;
import com.thundercomm.gateway.data.DeviceData;
import com.thundercomm.rtsp.LocalVideoDecoder;
import com.thundercomm.rtsp.TsIplayerStateListener;
import com.thundercomm.rtsp.TsRealTimePlayer;
import com.thundercomm.rtsp.TseDecodeType;
import com.thundercomm.rtsp.TseRtPlayState;


import org.greenrobot.eventbus.Subscribe;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Timer;
import java.util.TimerTask;

/**
 * play video.
 */
@SuppressLint("SetTextI18n")
public class PlayFragment extends Fragment implements TextureView.SurfaceTextureListener {

    private static final String TAG = PlayFragment.class.getSimpleName();

    protected String mUrl;

    private String name;

    private boolean isPreview;

    private TsRealTimePlayer mRealTimePlayer = null;

    public MyOnTouchListener myOnTouchListener;

    private int id;

    private SurfaceTexture surfaceTexture;

    private static final int TIMER_TASK_TIME = 1 * 25 * 1000;

    private static final int LOOP_TIME = 5 * 1000;

    private TextView nameTv;
    private TextView rtspInfoTv;
    private TextView rtspFpsTv;
    private TextView rtspLocationTv;
    private Timer timer;
    private Timer checkFaceTimer;
    private ReConnectTimerTask timerTask;
    private int warning_rect_width;
    private int warning_rect_height;
    boolean hasDrawn = false;
    private LocalVideoDecoder localVideoDecoder;
    SurfaceView mWrarningView;

    SurfaceHolder mWarningViewHolder;

    SurfaceView mFaceRectView;

    SurfaceHolder mFaceViewHolder;

    MyGLSurfaceView myGlSurfaceView;


    Paint paint_face;
    Paint paint_warning;
    Paint paint_frame;
    Paint paint_Txt;
    Paint paint_Txt_frame;
    TextPaint paint_Txt_Msg;
    TextPaint paint_Txt_Msg2;

    public PlayFragment(int id) {
        this.id = id;
    }

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EventBusManager.register(this);
    }


    /**
     * @Describe
     */
    Bitmap resizedBitmap = null;

    /**
     * 初始化画笔 用于绘制人脸识别框 物体识别框
     * @Describe
     */
    void initPaint() {
        paint_face = new Paint(Paint.ANTI_ALIAS_FLAG);
        paint_face.setColor(getResources().getColor(R.color.face_rect_color));
        paint_face.setShadowLayer(10f, 0, 0, Color.GREEN);
        paint_face.setStyle(Paint.Style.STROKE);
        paint_face.setStrokeWidth(4);
        paint_face.setFilterBitmap(true);


        paint_warning = new Paint();
        paint_warning.setColor(Color.RED);
        paint_warning.setStyle(Paint.Style.STROKE);
        paint_warning.setStrokeWidth(7);
        paint_warning.setFilterBitmap(true);
        paint_warning.setAlpha(120);
        // paint_warning.setShader(shader);

        paint_Txt = new Paint(Paint.ANTI_ALIAS_FLAG);
        paint_Txt.setARGB(255, 255, 0, 0);
        paint_Txt.setTextSize(100.0f);

        paint_Txt_Msg = new TextPaint(Paint.ANTI_ALIAS_FLAG);
        paint_Txt_Msg.setARGB(0xFF, 0xFF, 0, 0);
        paint_Txt_Msg.setTextSize(15.0F);

        paint_Txt_Msg2 = new TextPaint(Paint.ANTI_ALIAS_FLAG);
        paint_Txt_Msg2.setColor(Color.WHITE);
        paint_Txt_Msg2.setTextSize(15.0F);

        paint_frame = new Paint(Paint.ANTI_ALIAS_FLAG);
        paint_frame.setColor(Color.RED);
        paint_frame.setShadowLayer(10f, 0, 0, Color.RED);
        paint_frame.setStyle(Paint.Style.FILL);
        paint_frame.setStrokeWidth(4);
        paint_frame.setFilterBitmap(true);

        paint_Txt_frame = new Paint();
        paint_Txt_frame.setStyle(Paint.Style.FILL);
        paint_Txt_frame.setColor(Color.RED);
        paint_Txt_frame.setTextSize(40.0f);
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_playe, container, false);
        return view;
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        Jni.Affinity.bindToCpu(this.id % 4 + 4);
        nameTv = view.findViewById(R.id.tv_name);
        myGlSurfaceView = view.findViewById(R.id.fragment_gltv_view);
        myGlSurfaceView.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(SurfaceHolder holder) {
                LogUtil.e(TAG, "surfaceCreated");
                surfaceTexture = myGlSurfaceView.getSurfaceTexture();
                surfaceTexture.setOnFrameAvailableListener(frameAvailableListener);
                Surface surface = new Surface(surfaceTexture);
                initPlayer(surface);
            }

            @Override
            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                LogUtil.e(TAG, "surfaceChanged w:" + width + " h: " + height);
            }

            @Override
            public void surfaceDestroyed(SurfaceHolder holder) {
                LogUtil.e(TAG, "surfaceDestroyed");

            }
        });

        mFaceRectView = view.findViewById(R.id.fragment_rect_view);
        mFaceRectView.setZOrderMediaOverlay(true);
        mFaceViewHolder = mFaceRectView.getHolder();
        mWrarningView = view.findViewById(R.id.fragment_wrarning_view);
        mFaceRectView.setZOrderMediaOverlay(true);
        mWarningViewHolder = mWrarningView.getHolder();
        mWarningViewHolder.setFormat(PixelFormat.TRANSLUCENT);
        //getWarning_rect(mWarningViewHolder);
        //drawThread.start();
        mFaceViewHolder.addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(SurfaceHolder holder) {
                holder.setFormat(PixelFormat.TRANSPARENT);
            }

            @Override
            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {

            }

            @Override
            public void surfaceDestroyed(SurfaceHolder holder) {

            }
        });
        rtspInfoTv = view.findViewById(R.id.tv_RtspId);
        rtspFpsTv = view.findViewById(R.id.tv_rtsp_fps);
        rtspLocationTv = view.findViewById(R.id.tv_rtsp_location);

    }

    private void getWarning_rect(SurfaceHolder mFaceViewHolder) {
        Canvas canvas = null;
        synchronized (mFaceViewHolder) {
            try {
                canvas = this.mFaceViewHolder.lockCanvas();
                warning_rect_width = canvas.getWidth();
                warning_rect_height = canvas.getHeight();
                LogUtil.e(TAG, "get canvas" + warning_rect_width + " " + warning_rect_height);
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                if (null != canvas) {
                    this.mFaceViewHolder.unlockCanvasAndPost(canvas);
                }
            }
        }
    }

    TsIplayerStateListener mPlayerStateListener;

    private void initPlayer(Surface surface) {
        DeviceData deviceData = RtspItemCollection.getInstance().getDeviceList().get(this.id);
        getActivity().runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mUrl = RtspItemCollection.getInstance().getAttributesValue(deviceData, "URL");
                rtspInfoTv.setText("ID : " + mUrl);
            }
        });
        myGlSurfaceView.setSurfaceId(id);
        LogUtil.d("RealTimePlayer", "Init Player [url:" + mUrl + " id:" + id + " CameraName:" + name + "]");
        //rtspInfoTv.setText("id:" + id + " " + mUrl.split("@")[1]);
        if (null == this.mPlayerStateListener) {
            mPlayerStateListener = new TsIplayerStateListener() {
                @Override
                public void onStop() {
                    //Toast.makeText(getContext(), mUrl + "Error! stop", Toast.LENGTH_SHORT).show();
                    Log.d("RealTimePlayer", "onStop" + " -- [Url:" + mUrl + "]");
                    myGlSurfaceView.stopRendering();
                    if (myGlSurfaceView != null) {

                    }
                    Runtime.getRuntime().gc();
                }

                @Override
                public void onError(TseRtPlayState state) {
                    Log.d("RealTimePlayer", "TSIPlayerStateListener onError");
                    if (mRealTimePlayer != null) {
                        mRealTimePlayer.setPlayerListener(null);
                        mRealTimePlayer = null;
                        mRealTimePlayer.stop(id);
                    }

                }

                @Override
                public void onPrepareComplete() {
                    if (null != mRealTimePlayer) {
                        // start play video
                        mRealTimePlayer.start(mUrl, id, isPreview);
                    }
                    Log.d("RealTimePlayer", "onPrepareComplete");
                    long startTime = System.currentTimeMillis();
                    long endTime = System.currentTimeMillis();
                    boolean isStart = false;
                    while (endTime - startTime < LOOP_TIME) {
                        if (mRealTimePlayer != null) {
                            if (mRealTimePlayer.getVideoType() != null) {
                                if (!isStart) {

                                    isStart = true;
                                    if (timerTask != null) {
                                        timerTask.cancel();
                                    }
                                    if (timer != null) {
                                        timer.cancel();
                                    }
                                    timerTask = null;
                                    timer = null;
                                }
                                break;
                            }
                        }
                        endTime = System.currentTimeMillis();
                    }
                    if (!isStart) {
                        if (mRealTimePlayer != null) {
                            mRealTimePlayer.release(id);
                            mRealTimePlayer.setPlayerListener(null);
                            mRealTimePlayer = null;
                        }
                    }
                }

                @Override
                public void onPlay() {
                    Log.d("RealTimePlayer", "onPlay");
                }

            };
        }
        if (null == this.mRealTimePlayer) {
            this.mRealTimePlayer = new TsRealTimePlayer(getActivity());
        }
        this.mRealTimePlayer.setDecodeType(TseDecodeType.H264);
        this.mRealTimePlayer.setSurface(surface);
        this.mRealTimePlayer.setPlayerListener(mPlayerStateListener);
        this.mRealTimePlayer.prepare(id, isPreview);
    }

    public void startPlayAudio() {
        if (null != mRealTimePlayer) {
            this.mRealTimePlayer.setIsStartAudio(true);
            this.mRealTimePlayer.audioStart(id);
        }
    }

    public void stopPlayAudio() {
        if (null != mRealTimePlayer) {
            this.mRealTimePlayer.setIsStartAudio(false);
            this.mRealTimePlayer.stopAudio();
        }
    }

    @Override
    public void onDestroy() {
        if (localVideoDecoder != null) {
            localVideoDecoder.release();
            localVideoDecoder = null;
        }

        if (checkFaceTimer != null) {
            checkFaceTimer.cancel();
        }
        stopRealTimePlayer(null);
        super.onDestroy();
    }

    public void stopRealTimePlayer(@Nullable stopPlayerListener stopPlayerListener) {
        if (null != mRealTimePlayer) {
            if (mRealTimePlayer.isPlaying()) {
                mRealTimePlayer.stop(id);
            }
            mRealTimePlayer.release(id);
            mRealTimePlayer = null;
        }
        if (stopPlayerListener != null) {
            stopPlayerListener.onStop();
        }

    }

    @Override
    public void onConfigurationChanged(Configuration newConfig) {
        super.onConfigurationChanged(newConfig);
    }

    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture, int i, int i1) {
        Log.d(TAG, "onSurfaceTextureAvailable");
        this.surfaceTexture = surfaceTexture;
        initPlayer(new Surface(surfaceTexture));
    }

    @Override
    public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture, int i, int i1) {
        Log.d(TAG, "onSurfaceTextureSizeChanged");
    }

    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
        Log.d(TAG, "onSurfaceTextureDestroyed");
        // mSurfaceTexture = surfaceTexture;
        if (null != mRealTimePlayer) {
            if (mRealTimePlayer.isPlaying()) {
                mRealTimePlayer.stop(id);
            }
            mRealTimePlayer.release(id);
            mRealTimePlayer = null;
        }
        return false;
    }

    private long pts = System.currentTimeMillis();

    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {
        LogUtil.e(TAG, "onSurfaceTextureUpdated");
    }



    /**
     * rtsp重连任务
     * @Describe
     */
    private class ReConnectTimerTask extends TimerTask {

        @Override
        public void run() {
            if (null == mRealTimePlayer) {
                mRealTimePlayer = new TsRealTimePlayer(getActivity());
                Log.d("xuecq0313", "1");
            } else {
                Log.d("xuecq0313", "2");
                // mRealTimePlayer.release(id);
                String videoType = mRealTimePlayer.getVideoType();
                if (!TextUtils.isEmpty(videoType) && !videoType.equals("Err")) {
                    Log.d("xuecq0313", "reconnect success");

                    if (timerTask != null) {
                        timerTask.cancel();
                    }
                    if (timer != null) {
                        timer.cancel();
                    }
                    timerTask = null;
                    timer = null;
                    return;

                } else {
                    mRealTimePlayer.release(id);
                    mRealTimePlayer.setPlayerListener(null);
                    mRealTimePlayer = null;

                }
            }

            // mTextureView.setSurfaceTextureListener(PlayFragment.this);
            if (surfaceTexture != null) {
                Log.d("xuecq0313", "3");
                initPlayer(new Surface(surfaceTexture));
            } else {
                Log.d("xuecq0313", "surfaceTexture = null");
            }
            Log.d("xuecq0313", "mUrl = " + mUrl + " ,id = " + id);
        }
    }


    private SurfaceTexture.OnFrameAvailableListener frameAvailableListener = new SurfaceTexture.OnFrameAvailableListener() {
        @Override
        public void onFrameAvailable(SurfaceTexture surfaceTexture) {

        }
    };

    public class DrawThread extends Thread {
        private SurfaceHolder mHolder = null;
        private boolean isRun = false;
        private Rect faceRect;

        public DrawThread(SurfaceHolder holder) {
            LogUtil.d(TAG, "DrawThread Constructo");
            mHolder = holder;
        }

        public void setRun(boolean isRun) {
            LogUtil.d(TAG, "DrawThread setRun: " + isRun);
            this.isRun = isRun;
        }

        @Override
        public void run() {
            int x = 0;
            while (true) {
                try {
                    sleep(2000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                ArrayList<Rect> rects = new ArrayList<>();
                rects.add(new Rect(100, 100, 200, 200));
                //EventBusManager.postEvent(new FaceIdentificationResult(0, rects));
                x++;

            }
        }
    }




    @Subscribe
    public void updataFrameSpeed(EventMessage eventMessage) {
        if (eventMessage.getMessageType().equals("FRAME")) {
            if (eventMessage.getCameraId() == id) {
                if (getActivity() != null) {
                    getActivity().runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            int frame = (int) eventMessage.getObject();
                            rtspFpsTv.setText("ID:" + (id + 1) + "  1080@fps: " + frame);
                        }
                    });
                }
            }
        }
    }

    /**
     * 绘制一个自定义的框 --只绘制一个矩形的四个角
     *
     * @param x1     绘制矩形的x坐标
     * @param y1     绘制矩形的y坐标
     * @param w      绘制矩形的w宽
     * @param h      绘制矩形的h高
     * @param canvas 要绘制的目标画布
     * @param paint  使用何种画笔 画笔由{@link PlayFragment#initPaint()}方法生成 画笔：{@link PlayFragment#paint_face},{@link PlayFragment#paint_warning},{@link PlayFragment#paint_Txt}
     */
    void drawRound(Canvas canvas, int x1, int y1, int w, int h, Paint paint) {
        int x = x1;
        int y = y1;
        int right = x + w;
        int bottom = y + h;
        float lineLength_x = (float) (w / 3.5);
        float lineLength_y = (float) (h / 3.5);
        //左上角
        canvas.drawLine(x, y, x, y + lineLength_y, paint);
        canvas.drawLine(x, y, x + lineLength_x, y, paint);
        //左下角
        canvas.drawLine(x, bottom, x + lineLength_x, bottom, paint);
        canvas.drawLine(x, bottom, x, bottom - lineLength_y, paint);
        //右下角
        canvas.drawLine(right, bottom, right - lineLength_x, bottom, paint);
        canvas.drawLine(right, bottom, right, bottom - lineLength_y, paint);
        //右上角
        canvas.drawLine(right, y, right - lineLength_x, y, paint);
        canvas.drawLine(right, y, right, y + lineLength_y, paint);
    }

    Point get_show_coordinate(int window_width, int window_height, Rect rect) {
        int bottom = rect.bottom;
        int top = rect.top;
        int left = rect.left;
        int right = rect.right;

        int new_y = top;
        int new_x = (int) ((rect.right - rect.left) / 3.5) + rect.left;
        Point newPoint = new Point();
        if (top - 10 < 20) {
            //框靠近上边界，文字位置放在下边界
            new_y = rect.bottom + 25;
        } else if (window_height - bottom < 20) {
            //框靠近下边界，文字位置放在下边界
            new_y = top - 10;
        } else {
            new_y = top - 5;
        }
        newPoint.set(new_x, new_y);
        return newPoint;
    }

    /**
     * 清空显示框的方法 暴露给外部调用 调用此方法会调用{@link PlayFragment#clearRects(SurfaceHolder)}
     *
     * @Describe
     */
    public void OnClean() {
        clearObjectRect();
    }

    private void clearObjectRect() {
        if (hasDrawn) {
            clearRects(mFaceViewHolder);
        }

    }

    /**
     * 清空画布
     *
     * @param mHolder 要清空的画布的holder
     * @Describe
     */
    private void clearRects(final SurfaceHolder mHolder) {
        if (mHolder != null) {
            synchronized (mHolder) {
                Canvas canvas = mHolder.lockCanvas();
                if (canvas != null) {
                    canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
                    mHolder.unlockCanvasAndPost(canvas);
                }
            }
        }

    }


    private int findDeviceByName(String device) {
        return DeviceCollection.getInstance().findDeviceByName(device);
    }


    private class FaceRects {
        private ArrayList<Rect> rectArrayList;
        private long pts;

        public ArrayList<Rect> getRectArrayList() {
            return rectArrayList;
        }

        public void setRectArrayList(ArrayList<Rect> rectArrayList) {
            this.rectArrayList = rectArrayList;
        }

        public long getPts() {
            return pts;
        }

        public void setPts(long pts) {
            this.pts = pts;
        }

        public FaceRects(ArrayList<Rect> rectArrayList, long pts) {
            this.rectArrayList = rectArrayList;
            this.pts = pts;
        }

        @Override
        public String toString() {
            return "FaceRects{" +
                    "rectArrayList=" + rectArrayList +
                    ", pts=" + pts +
                    '}';
        }
    }

    public interface MyOnTouchListener {
        void onTouch(MotionEvent event, int index);
    }

    public void setMyOnTouchListener(MyOnTouchListener myOnTouchListener) {
        this.myOnTouchListener = myOnTouchListener;
    }

    public int getWarning_rect_width() {
        return warning_rect_width;
    }

    public int getWarning_rect_height() {
        return warning_rect_height;
    }

    public void show(AppCompatActivity activity) {
        FragmentTransaction ft = activity.getSupportFragmentManager().beginTransaction();
        ft.add(this, this.getClass().getSimpleName());
        ft.commitAllowingStateLoss();//注意这里使用commitAllowingStateLoss()
    }

    @Override
    public void onSaveInstanceState(@NonNull Bundle outState) {
    }

    public interface stopPlayerListener {
        void onStop();
    }
}

