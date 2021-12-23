package com.thundercomm.eBox.AI;


import com.thundercomm.eBox.Config.VisionTestConfig;

import android.app.Application;
import android.graphics.Bitmap;
import android.media.Image;
import android.util.Log;
import android.view.Gravity;
import android.widget.Toast;

import com.thundercomm.eBox.Constants.Constants;
import com.thundercomm.eBox.Jni;
import com.thundercomm.eBox.Model.RtspItemCollection;
import com.thundercomm.eBox.Utils.LogUtil;
import com.thundercomm.eBox.VIew.GestureClassificationFragment;
import com.thundercomm.eBox.VIew.PlayFragment;
import com.thundercomm.gateway.data.DeviceData;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;


import java.io.BufferedReader;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;

import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.StringTokenizer;
import java.util.Vector;

import lombok.SneakyThrows;


public class GestureClassificationTask {

    private static String TAG = "GestureClassificationTask";

    private HashMap<Integer, Interpreter> mapGestureClassification = new HashMap<Integer, Interpreter>();

    private HashMap<Integer, DataInputFrame> inputFrameMap = new HashMap<Integer, DataInputFrame>();
    private Vector<GestureClassificationTaskThread> mGestureClassificationTaskThreads = new Vector<GestureClassificationTaskThread>();

    private boolean istarting = false;
    private boolean isInit = false;
    private Application mContext;
    private ArrayList<PlayFragment> playFragments;

    private int frameWidth;
    private int frameHeight;

    private static volatile GestureClassificationTask _instance;

    private GestureClassificationTask() {
    }

    public static GestureClassificationTask getGestureClassificationTask() {
        if (_instance == null) {
            synchronized (GestureClassificationTask.class) {
                if (_instance == null) {
                    _instance = new GestureClassificationTask();
                }
            }
        }
        return _instance;
    }

    public void init( Application context, Vector<Integer> idlist, ArrayList<PlayFragment> playFragments, int width, int height) {
        LogUtil.d(TAG, "init AI");
        frameWidth = width;
        frameHeight = height;
        interrupThread();
        for (int i = 0; i < idlist.size(); i++) {
            if (getGestureClassificationAlgorithmType(idlist.elementAt(i))) {
                DataInputFrame data = new DataInputFrame(idlist.elementAt(i));
                inputFrameMap.put(idlist.elementAt(i), data);
            }
        }
        mContext = context;
        istarting = true;
        isInit = true;
        this.playFragments = playFragments;
        Log.d("", "dlist.size():"+ idlist.size());
        for (int i = 0; i < idlist.size(); i++) {
            if (getGestureClassificationAlgorithmType(idlist.elementAt(i))) {
                GestureClassificationTaskThread gestureClassificationTaskThread = new GestureClassificationTaskThread(idlist.elementAt(i));
                gestureClassificationTaskThread.start();
                mGestureClassificationTaskThreads.add(gestureClassificationTaskThread);
            }
        }
    }

    private boolean getGestureClassificationAlgorithmType(int id) {
        DeviceData deviceData = RtspItemCollection.getInstance().getDeviceList().get(id);
        boolean enable = Boolean.parseBoolean(RtspItemCollection.getInstance().getAttributesValue(deviceData, Constants.ENABLE_GESTURECLASSIFICATION_STR));
        return enable;
    }

    public void addImgById(int id, final Image img) {
        if (!inputFrameMap.containsKey(id)) {
            return;
        }

        DataInputFrame data = inputFrameMap.get(id);
        data.addImgById(img);
    }

    public void addBitmapById(int id, final Bitmap bmp, int w, int h) {
        if (!inputFrameMap.containsKey(id)) {
            return;
        }

        DataInputFrame data = inputFrameMap.get(id);
        data.org_w = w;
        data.org_h = h;
        data.addBitMapById(bmp);
    }

    public void addMatById(int id, final Mat img, int w, int h) {
        if (!inputFrameMap.containsKey(id)) {
            return;
        }

        DataInputFrame data = inputFrameMap.get(id);
        data.org_w = w;
        data.org_h = h;
        data.addMatById(img);
    }


    class GestureClassificationTaskThread extends Thread {

        private GestureClassificationFragment gestureClassificationTask = null;

        private static final String TF_API_MODEL_FILE = "model.tflite";
        private static final String TF_API_LABELS_FILE = "labels.txt";

        // The inception net requires additional normalization of the used input.
        private static final float IMAGE_MEAN = 1.0f;

        private static final float IMAGE_STD = 127.0f;

        // An array to hold inference results, to be feed into Tensorflow Lite as outputs.
        private float[][] labelProbArray;

        // multi-stage low pass filter
        private float[][] filterLabelProbArray;

        private static final int FILTER_STAGES = 3;
        private static final float FILTER_FACTOR = 0.4f;

        private static final float THRESHOLD = 0.1f;

        private static final int RESULTS_TO_SHOW = 3;
        private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
                new PriorityQueue<>(
                        RESULTS_TO_SHOW,
                        new Comparator<Map.Entry<String, Float>>() {
                            @Override
                            public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                                return (o1.getValue()).compareTo(o2.getValue());
                            }
                        });

        // Number of threads in the java app
        private static final int NUM_THREADS = 4;

        // Config values.
        private final int inputSize = 224;;
        // Pre-allocated buffers.
        private final List<String> labels = new ArrayList<>();
        private int[] intValues;

        private ByteBuffer imgData;

        private Interpreter tfLite;
        int alg_camid = -1;

        private boolean isDuringChecking;
        private long nowTime = 0;
        private long startTime = 0;
        private boolean alreadyError = false;

        private HashMap<String, Integer> mapGestureCount = new HashMap<String, Integer>();
        private int index_E = 0;



        GestureClassificationTaskThread(int id) {
            alg_camid = id;
            imgData = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4);
            imgData.order(ByteOrder.nativeOrder());
            intValues = new int[inputSize * inputSize];
            if (!mapGestureClassification.containsKey(alg_camid)) {
                try {
                    MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(mContext, TF_API_MODEL_FILE);

                    BufferedReader reader =
                            new BufferedReader(new InputStreamReader(mContext.getAssets().open(TF_API_LABELS_FILE)));
                    String line;
                    line = reader.readLine();
                    reader.close();

                    StringTokenizer tokenizer = new StringTokenizer(line, ",");
                    while (tokenizer.hasMoreTokens()) {
                        String token = tokenizer.nextToken();
                        labels.add(token);
                    }

                    Interpreter.Options options = new Interpreter.Options();
                    options.setNumThreads(NUM_THREADS);
                    options.setUseXNNPACK(true);
                    options.setUseNNAPI(true);
                    tfLite = new Interpreter(tfliteModel, options);
                } catch (Exception e) {
                    e.printStackTrace();
                }

                mapGestureClassification.put(alg_camid, tfLite);
            } else {
                tfLite = mapGestureClassification.get(alg_camid);
            }

            labelProbArray = new float[1][labels.size()];
            filterLabelProbArray = new float[FILTER_STAGES][labels.size()];
        }

        @SneakyThrows
        @Override
        public void run() {
            super.run();
            Jni.Affinity.bindToCpu(alg_camid % 4 + 4);
            gestureClassificationTask = (GestureClassificationFragment) playFragments.get(alg_camid);
            DataInputFrame inputFrame = inputFrameMap.get(alg_camid);
            Mat rotateimage = new Mat(frameHeight, frameWidth, CvType.CV_8UC4);
            Mat resizeimage = new Mat(frameHeight, frameWidth, CvType.CV_8UC4);
            Mat frameBgrMat = new Mat(frameHeight, frameWidth, CvType.CV_8UC3);
            LogUtil.d("", "debug test start camid  " + alg_camid);
            showToast("Please make gestures according to the direction of \"E\" in the upper left"
                    + " corner", true);
            sleep(2000);
            while (istarting) {
                try {
                    inputFrame.updateFaceRectCache();
                    Mat mat = inputFrame.getMat();

                    if (!OPencvInit.isLoaderOpenCV() || mat == null) {
                        if (mat != null) mat.release();
                        try {
                            Thread.sleep(50);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                        continue;
                    }

                    Core.flip(mat, rotateimage, 0);
                    Imgproc.resize(rotateimage, resizeimage, new Size(frameHeight, frameWidth));
                    Imgproc.cvtColor(resizeimage, frameBgrMat, Imgproc.COLOR_BGRA2BGR);
                    Bitmap bitmap = Bitmap.createBitmap(frameBgrMat.width(), frameBgrMat.height(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(frameBgrMat, bitmap);
                    imgData.rewind();
                    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
                    for (int i = 0; i < inputSize; ++i) {
                        for (int j = 0; j < inputSize; ++j) {
                            int pixelValue = intValues[i * inputSize + j];

                            imgData.putFloat((((pixelValue >> 16) & 0xFF) / IMAGE_STD) - IMAGE_MEAN);
                            imgData.putFloat((((pixelValue >> 8) & 0xFF) / IMAGE_STD) - IMAGE_MEAN);
                            imgData.putFloat(((pixelValue & 0xFF) / IMAGE_STD) - IMAGE_MEAN);
                        }
                    }
                    int numLabels = labels.size();

                    // Run the inference call.
                    tfLite.run(imgData, labelProbArray);

                    for (int j = 0; j < numLabels; ++j) {
                        filterLabelProbArray[0][j] +=
                                FILTER_FACTOR * (getProbability(j) - filterLabelProbArray[0][j]);
                    }
                    for (int i = 1; i < FILTER_STAGES; ++i) {
                        for (int j = 0; j < numLabels; ++j) {
                            filterLabelProbArray[i][j] +=
                                    FILTER_FACTOR * (filterLabelProbArray[i - 1][j] - filterLabelProbArray[i][j]);
                        }
                    }

                    for (int j = 0; j < numLabels; ++j) {
                        setProbability(j, filterLabelProbArray[FILTER_STAGES - 1][j]);
                    }

                    getTopKLabels();

                } catch (final Exception e) {
                    e.printStackTrace();
                    LogUtil.e(TAG, "Exception!");
                }
            }
        }

        private void postNoResult() {
            if (gestureClassificationTask != null) {
                gestureClassificationTask.OnClean();
            }
        }

        private void postDetectResult(String label, int index_E) {
            if (gestureClassificationTask != null) {
                gestureClassificationTask.onDraw(label, index_E);
            }
        }

        private void drawE(int index_E) {
            if (gestureClassificationTask != null) {
                gestureClassificationTask.onDrawE(index_E);
            }
        }

        private float getProbability(int labelIndex) {
            return labelProbArray[0][labelIndex];
        }

        private void setProbability(int labelIndex, Number value) {
            labelProbArray[0][labelIndex] = value.floatValue();
        }

        private float getNormalizedProbability(int labelIndex) {
            return getProbability(labelIndex);
        }

        private void getTopKLabels() {
            for (int i = 0; i < labels.size(); ++i) {
                sortedLabels.add(
                        new AbstractMap.SimpleEntry<>(labels.get(i), getNormalizedProbability(i)));
                if (sortedLabels.size() > RESULTS_TO_SHOW) {
                    sortedLabels.poll();
                }
            }
            final int size = sortedLabels.size();
            for (int i = 0; i < size; i++) {
                Map.Entry<String, Float> label = sortedLabels.poll();
                if (i == size - 1) {
                    if (label.getValue() > THRESHOLD) {
                        if(isDuringChecking) {
                            nowTime = System.currentTimeMillis();
                            if (nowTime - startTime >= 4 * 1000) {
                                mapGestureCount.put(label.getKey(), mapGestureCount.get(label.getKey()) != null ?
                                        mapGestureCount.get(label.getKey()) + 1 : 1);
                                String maxCountKey = "";
                                int maxCountValue = 0;
                                for(Map.Entry<String, Integer> entry : mapGestureCount.entrySet()){
                                    if (entry.getValue() > maxCountValue) {
                                        maxCountKey = entry.getKey();
                                    }
                                }
                                if (maxCountKey.equals(VisionTestConfig.E_direct[index_E])) {
                                    postDetectResult(VisionTestConfig.E_direct[index_E], index_E);
                                    if(alreadyError) {
                                        alreadyError = false;
                                    }
                                    showToast("Pass!", true);
                                    sleep(2000);
                                } else if (!alreadyError) {
                                    postDetectResult(maxCountKey, index_E);
                                    showToast("Error! Please confirm again", false);
                                    alreadyError = true;
                                    index_E--;
                                    sleep(2000);
                                } else {
                                    postDetectResult(maxCountKey, index_E);
                                    showToast("This round of testing is over! The vision of your eye is:" + VisionTestConfig.E_score[index_E], false);
                                    alreadyError = false;
                                    closeService();
                                    sleep(2000);
                                    postNoResult();
                                }

                                if (index_E == VisionTestConfig.E_direct.length - 1) {
                                    showToast("This round of testing is over! The vision of your eye is:" + VisionTestConfig.E_score[index_E], false);
                                    postNoResult();
                                    closeService();
                                }
                                index_E++;
                                isDuringChecking = false;
                                startTime = nowTime;
                                mapGestureCount.clear();
                            } else {
                                mapGestureCount.put(label.getKey(), mapGestureCount.get(label.getKey()) != null ?
                                        mapGestureCount.get(label.getKey()) + 1 : 1);
                                drawE(index_E);
                            }
                        } else {
                            isDuringChecking = true;
                            nowTime = System.currentTimeMillis();
                            startTime = nowTime;
                            mapGestureCount.put(label.getKey(), mapGestureCount.get(label.getKey()) != null ?
                                    mapGestureCount.get(label.getKey()) + 1 : 1);
                            drawE(index_E);
                        }

                    }
                }
            }
        }
        private void sleep(int time_ms) {
            try {
                Thread.sleep(time_ms);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        private void showToast(String message, boolean short_display) {
            gestureClassificationTask.getActivity().runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    int duration = short_display ? Toast.LENGTH_SHORT : Toast.LENGTH_LONG;
                    Toast toast = Toast.makeText(gestureClassificationTask.getContext(), message, duration);
                    toast.setGravity(Gravity.CENTER_VERTICAL, 0, 0);
                    toast.show();
                }
            });
        }

    }



    public void closeService() {
        isInit = false;
        istarting = false;

        System.gc();
        System.gc();
    }

    private void interrupThread() {
        for (GestureClassificationTaskThread multiObjectDetectionTaskThread : this.mGestureClassificationTaskThreads) {
            if (multiObjectDetectionTaskThread != null && !multiObjectDetectionTaskThread.isInterrupted()) {
                multiObjectDetectionTaskThread.interrupt();
            }
        }
        mapGestureClassification.clear();
    }

    public boolean isIstarting() {
        return isInit;
    }
}
