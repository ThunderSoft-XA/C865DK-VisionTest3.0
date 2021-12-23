package com.thundercomm.eBox.Gl;

import android.content.Context;
import android.graphics.SurfaceTexture;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLException;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.util.AttributeSet;


import com.thundercomm.eBox.AI.OPencvInit;
import com.thundercomm.eBox.AI.GestureClassificationTask;
import com.thundercomm.eBox.Constants.Constants;
import com.thundercomm.eBox.Model.RtspItemCollection;
import com.thundercomm.eBox.Utils.LogUtil;
import com.thundercomm.gateway.data.DeviceData;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLDisplay;
import javax.microedition.khronos.opengles.GL10;


class MyConfigChooser implements GLSurfaceView.EGLConfigChooser {
    @Override
    public EGLConfig chooseConfig(EGL10 egl,
                                  EGLDisplay display) {

        int attribs[] = {
                EGL10.EGL_LEVEL, 0,
                EGL10.EGL_RENDERABLE_TYPE, 4,  // EGL_OPENGL_ES2_BIT
                EGL10.EGL_COLOR_BUFFER_TYPE, EGL10.EGL_RGB_BUFFER,
                EGL10.EGL_RED_SIZE, 8,
                EGL10.EGL_GREEN_SIZE, 8,
                EGL10.EGL_BLUE_SIZE, 8,
                EGL10.EGL_DEPTH_SIZE, 16,
                EGL10.EGL_SAMPLE_BUFFERS, 1,
                EGL10.EGL_SAMPLES, 4,  // 在这里修改MSAA的倍数，4就是4xMSAA，再往上开程序可能会崩
                EGL10.EGL_NONE
        };
        EGLConfig[] configs = new EGLConfig[1];
        int[] configCounts = new int[1];
        egl.eglChooseConfig(display, attribs, configs, 1, configCounts);

        if (configCounts[0] == 0) {
            // Failed! Error handling.
            return null;
        } else {
            return configs[0];
        }
    }

}

public class MyGLSurfaceView extends GLSurfaceView {
    private static final String TAG = "MyGLSurfaceView";
    MyRenderer renderer;

    int screenWidth, screenHeight;
    boolean playing;
    private static boolean ISTAKEPHOTO = true;
    private int surfaceId;
    private int mOutputWidth;
    private int mOutputHeight;
    private int mvpMatrixLoc;
    private int algorithmType = -1;
    private PutIntoTask p;
    private boolean allowRender = false;

    public MyGLSurfaceView(Context context, AttributeSet attri) {
        super(context, attri);
        init(context);
    }

    public MyGLSurfaceView(Context context) {
        super(context);
        init(context);
    }

    private void init(Context context) {
        //this.context = (MainActivity1) context;
        setEGLContextClientVersion(3);
        setEGLConfigChooser(new MyConfigChooser());
        renderer = new MyRenderer();
        allowRender = true;
        setRenderer(renderer);
        setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
        playing = false;
    }

    public SurfaceTexture getSurfaceTexture() {
        return renderer.createSurfaceTexture();
    }

    public void stopRendering() {
        renderer.release();
        renderer = null;
    }


    class MyRenderer implements Renderer {
        private static final String TAG = "MyRenderer";
        private int textureId;
        private SurfaceTexture mSurfaceTexture;

        private String vertShader;
        private String fragShader_Pre;
        private int programHandle;
        private int mPositionHandle;
        private int mTextureCoordHandle;

        FloatBuffer verticesBuffer, textureVerticesPreviewBuffer;
        private ShortBuffer drawListBuffer;

        // number of coordinates per vertex in this array
        private final int COORDS_PER_VERTEX = 2;

        private final int vertexStride = COORDS_PER_VERTEX * 4; // 4 bytes per vertex

        private final float squareVertices[] = { // in counterclockwise order:
                -1.0f, 1.0f,
                -1.0f, -1.0f,
                1.0f, -1.0f,
                1.0f, 1.0f
        };
        private final float textureVerticesPreview[] = { // in counterclockwise order:
                0.0f, 1.0f,
                1.0f, 1.0f,
                1.0f, 0.0f,
                0.0f, 0.0f
        };
        private final short drawOrder[] = {0, 1, 2, 0, 2, 3}; // order to draw vertices

        @Override
        public void onSurfaceCreated(GL10 gl, EGLConfig config) {

            mOutputWidth = screenWidth;
            mOutputHeight = screenHeight;
            LogUtil.e(TAG, "onSurfaceCreated");
            GLES20.glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
            initTexture();
            getAlgorithmType();
            LogUtil.e(TAG, "getAlgorithmType " + algorithmType);
        }

        int[] fbo;

        @Override
        public void onSurfaceChanged(GL10 gl, int width, int height) {
            computeMatrix();
            screenWidth = width;
            screenHeight = height;
            mOutputWidth = screenWidth;
            mOutputHeight = screenHeight;
            LogUtil.e(TAG, "onSurfaceChanged " + "w :" + width, "h: " + height);
            GLES20.glViewport(0, 0, width, height);
        }


        @Override
        public void onDrawFrame(final GL10 arg0) {

            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);
            if (mSurfaceTexture != null) {
                mSurfaceTexture.updateTexImage();
                //绘制纹理
                draw();
                //使用策略模式送算法.
                if (p != null) {
                    p.putInTask(getSurfaceId(), getMatFromGlSurface(0, 0, mOutputWidth,
                            mOutputHeight, arg0), arg0);
                }
            }
        }


        float[] modelMatrix = new float[16];

        private void computeMatrix() {
            long pts = System.currentTimeMillis();
            Matrix.setIdentityM(modelMatrix, 0);
            Matrix.rotateM(modelMatrix, 0, 90f, 0f, 0f, 1f);
            LogUtil.e(TAG, "rotateM takeTime" + (System.currentTimeMillis() - pts));
        }


        SurfaceTexture createSurfaceTexture() {
            textureId = createVideoTexture();
            mSurfaceTexture = new SurfaceTexture(textureId);
            return mSurfaceTexture;
        }

        private int createVideoTexture() {
            int[] texture = new int[1];

            GLES20.glGenTextures(1, texture, 0);
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, texture[0]);
            GLES20.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                    GL10.GL_TEXTURE_MIN_FILTER, GL10.GL_LINEAR);
            GLES20.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                    GL10.GL_TEXTURE_MAG_FILTER, GL10.GL_LINEAR);
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                    GL10.GL_TEXTURE_WRAP_S, GLES20.GL_NEAREST_MIPMAP_LINEAR);
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                    GL10.GL_TEXTURE_WRAP_T, GL10.GL_NEAREST_MIPMAP_LINEAR);

            return texture[0];
        }


        void initTexture() {
            fbo = new int[1];
            verticesBuffer = IVCGLLib.glToFloatBuffer(squareVertices);
            textureVerticesPreviewBuffer = IVCGLLib
                    .glToFloatBuffer(textureVerticesPreview);
            drawListBuffer = IVCGLLib.glToShortBuffer(drawOrder);

            vertShader = IVCGLLib.loadFromAssetsFile(
                    "IVC_VShader_Preview.sh", getContext().getResources());
            fragShader_Pre = IVCGLLib.loadFromAssetsFile(
                    "IVC_FShader_Preview.sh", getContext().getResources());

            programHandle = IVCGLLib.glCreateProgram(vertShader, fragShader_Pre);
            mPositionHandle = GLES20.glGetAttribLocation(programHandle, "position");
            mTextureCoordHandle = GLES20.glGetAttribLocation(programHandle, "inputTextureCoordinate");
            mvpMatrixLoc = GLES20.glGetUniformLocation(programHandle, "mvpMatrix");
        }



        void draw() {
            IVCGLLib.glUseFBO(0, 0, screenWidth, screenHeight, true, fbo[0], 0);
            //long pts = System.currentTimeMillis();
            GLES20.glUseProgram(programHandle);
            GLES20.glEnableVertexAttribArray(mPositionHandle);
            GLES20.glVertexAttribPointer(mPositionHandle, COORDS_PER_VERTEX, GLES20.GL_FLOAT, false, vertexStride, verticesBuffer);
            GLES20.glEnableVertexAttribArray(mTextureCoordHandle);
            GLES20.glVertexAttribPointer(mTextureCoordHandle, COORDS_PER_VERTEX, GLES20.GL_FLOAT, false, vertexStride, textureVerticesPreviewBuffer);

            GLES20.glEnable(GLES20.GL_BLEND);
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);
            GLES20.glUniform1i(GLES20.glGetUniformLocation(programHandle, "sampler2d"), 0);

            GLES20.glUniformMatrix4fv(mvpMatrixLoc, 1, false, modelMatrix, 0);
            GLES20.glDrawElements(GLES20.GL_TRIANGLE_STRIP, drawOrder.length, GLES20.GL_UNSIGNED_SHORT, drawListBuffer);
            IVCGLLib.glCheckGlError("glDrawElements");
            GLES20.glDisable(GLES20.GL_BLEND);
            GLES20.glDisableVertexAttribArray(mPositionHandle);
            GLES20.glDisableVertexAttribArray(mTextureCoordHandle);
            //LogUtil.e(TAG, "rotateM takeTime " + (System.currentTimeMillis() - pts) + "ms");
        }

        void release() {

            LogUtil.e(TAG, "deleting program " + programHandle);
            GLES20.glDeleteProgram(programHandle);
            programHandle = -1;

            LogUtil.e(TAG, "releasing SurfaceTexture");
            if (mSurfaceTexture != null) {
                mSurfaceTexture.release();
                mSurfaceTexture = null;
            }
        }
    }

    /**
     * 判断应该送什么算法
     *
     * @param id      surfaceID
     * @param typeKey 对应算法键值
     * @Describe 判断应该送什么算法
     */
    private boolean getAlgorithmType(int id, String typeKey) {
        DeviceData deviceData = RtspItemCollection.getInstance().getDeviceList().get(id);
        boolean enable = Boolean.parseBoolean(RtspItemCollection.getInstance().getAttributesValue(deviceData, typeKey));
        return enable;
    }

    /**
     * 判断应该送什么算法
     *
     * @Describe 判断应该送什么算法
     */
    private void getAlgorithmType() {
        if (getAlgorithmType(surfaceId, Constants.ENABLE_GESTURECLASSIFICATION_STR)) {
            p = new PutIntoMultiObjectDetectionTask();
        } else {
            p = null;
        }
    }

    private void getObjectInfo(int surfaceId, Mat mat) {
        //todo: 上传
        GestureClassificationTask.getGestureClassificationTask().addMatById(surfaceId, mat, mOutputWidth, mOutputHeight);
    }

    private ByteBuffer matPixelBuffer = null;
    private byte[] matData = null;

    /**
     * 获取glsurface上的cv：Mat数据 获取cvtype为CvType.CV_8UC4
     *
     * @param x  获取位置x坐标
     * @param y  获取位置y坐标
     * @param w  获取图像宽 w
     * @param h  获取图像高 h
     * @param gl 获取图像的opengl gl
     * @Describe
     */
    private Mat getMatFromGlSurface(int x, int y, final int w, final int h, GL10 gl) {
        if (matPixelBuffer == null) {
            matPixelBuffer = ByteBuffer.allocateDirect(w * h * 4);
            matPixelBuffer.order(ByteOrder.nativeOrder());
        }
        matPixelBuffer.clear();

        Mat mat = null;
        try {
            gl.glReadPixels(x, y, w, h, GL10.GL_RGBA, GL10.GL_UNSIGNED_BYTE,
                    matPixelBuffer);
            //long per_start_time = System.currentTimeMillis();
            if (matData == null) {
                matData = new byte[matPixelBuffer.capacity()];
            }
            ((ByteBuffer) matPixelBuffer.duplicate().clear()).get(matData);
            mat = new Mat(h, w, CvType.CV_8UC4);
            mat.put(0, 0, matData);
            //LogUtil.d(TAG, "camid "+ getSurfaceId() + " debug test add sendMatFromGLSurface per time " + (System.currentTimeMillis() - per_start_time));
        } catch (GLException e) {
            e.printStackTrace();
        }
        return mat;
    }

    public int getSurfaceId() {
        return surfaceId;
    }

    public void setSurfaceId(int surfaceId) {
        this.surfaceId = surfaceId;
    }

    /**
     * 送算法接口 不同的送算法方式通过实现这个接口完成送不同的算法
     *
     * @Describe
     */
    private interface PutIntoTask {
        void putInTask(int surfaceId, Mat matFromGlSurface, GL10 gl);
    }


    private class PutIntoMultiObjectDetectionTask implements PutIntoTask {
        private long pts = 0;

        @Override
        public void putInTask(int surfaceId, Mat matFromGlSurface, GL10 arg0) {
            if (GestureClassificationTask.getGestureClassificationTask().isIstarting()) {
                if (System.currentTimeMillis() - pts >= Constants.INPUT_GESTURECLASSIFICATION_INTERVAL) {
                    if (OPencvInit.isLoaderOpenCV()) {
                        getObjectInfo(getSurfaceId(), getMatFromGlSurface(0, 0, mOutputWidth,
                                mOutputHeight, arg0));
                    }
                    pts = System.currentTimeMillis();
                }
            }
        }
    }



}
