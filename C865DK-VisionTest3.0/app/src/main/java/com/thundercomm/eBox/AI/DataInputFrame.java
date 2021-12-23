package com.thundercomm.eBox.AI;

import android.graphics.Bitmap;
import android.graphics.Rect;
import android.media.Image;

import org.opencv.core.Mat;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.concurrent.ArrayBlockingQueue;

public class DataInputFrame {

    public int camId;
    public int org_w = 0;
    public int org_h = 0;
    private int MAX_DTATAS = 5;
    private int MAX_LIMIT = 3;

    private ArrayDeque<Image> mImageDatas = new ArrayDeque<Image>(MAX_DTATAS);
    private ArrayBlockingQueue<Bitmap> mBitmapDatas = new ArrayBlockingQueue<Bitmap>(MAX_DTATAS);
    private ArrayBlockingQueue<Mat> mMatDatas = new ArrayBlockingQueue<Mat>(MAX_DTATAS);

    private ArrayList<FaceRectCache> mFaceCaches = new ArrayList<FaceRectCache>(1000);

    private class FaceRectCache {
        long stamp;
        int maxD = 80 * 80;
        Rect rect;

        FaceRectCache(Rect rect) {
            stamp = System.currentTimeMillis();
            this.rect = rect;
        }

        boolean isNew(FaceRectCache comp) {
            boolean ret = false;
            int centerX0 = this.rect.centerX();
            int centerY0 = this.rect.centerY();
            int centerX1 = comp.rect.centerX();
            int centerY1 = comp.rect.centerY();
            double juli = Math.abs((centerX0 - centerX1) * (centerX0 - centerX1) + (centerY0 - centerY1) * (centerY0 - centerY1));
            if (juli > maxD) {
                ret = true;
            }
            if (Math.abs(comp.stamp - this.stamp) > 30 * 1000
                    && juli <= maxD) {
                ret = true;
            }
            return ret;
        }
    }

    synchronized boolean isNewFaceRectCache(Rect rect) {
        boolean ret = false;
        FaceRectCache cache = new FaceRectCache(rect);
        int size = mFaceCaches.size();
        int i = 0;
        for (i = 0; i < size; i++) {
            FaceRectCache c = mFaceCaches.get(i);
            if (c.isNew(cache)) {
                ret = true;
            }
        }
        if (i < size) {
            mFaceCaches.set(i, cache);
        } else if (i >= size) {
            ret = true;
            mFaceCaches.add(cache);
        }
        return ret;
    }

    public synchronized void updateFaceRectCache() {
        Iterator<FaceRectCache> it = mFaceCaches.iterator();
        while (it.hasNext()) {
            FaceRectCache c = it.next();
            if (System.currentTimeMillis() - c.stamp >= 30 * 1000) {
                it.remove();
            }
        }
    }

    public DataInputFrame(int id) {
        camId = id;
    }

    public void addImgById(Image img) {
        if (mImageDatas.size() >= MAX_LIMIT) {
            mImageDatas.poll();
            img.close();
        }
        if (img == null) return;

        mImageDatas.add(img);
    }

    public Image getImg() {
        return mImageDatas.poll();
    }

    public void addBitMapById(Bitmap bmp) {
        try {
            if (mBitmapDatas.size() >= MAX_LIMIT) {
                Bitmap tmp = mBitmapDatas.take();
                if (tmp != null && !tmp.isRecycled()) {
                    // TODO release bmp
                }
            }
            if (bmp == null) return;
            mBitmapDatas.put(bmp);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public Bitmap getBitMap() {
        Bitmap bmp = null;
        try {
            bmp = mBitmapDatas.take();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return bmp;
    }

    public void addMatById(Mat mat) {
        while (mMatDatas.size() >= MAX_LIMIT) {
            Mat tmp = null;
            try {
                tmp = mMatDatas.take();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            if (tmp != null) {
                tmp.release();
            }
        }
        if (mat == null) return;
        try {
            mMatDatas.put(mat);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        //LogUtil.d(TAG, "camid " + camId + " debug get mat list run per time add");
    }

    public Mat getMat() {
        Mat mat = null;
        try {
            mat = mMatDatas.take();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return mat;
    }
}
