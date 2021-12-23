package com.thundercomm.eBox.Utils;

import android.app.Activity;
import android.content.Context;
import android.content.res.Configuration;
import android.graphics.Point;
import android.util.DisplayMetrics;
import android.view.Display;
import android.view.WindowManager;

public class WindowUtils {
    private static String TAG = "WindowUtils--->";

    public static int getSurfaceViewWidth() {
        return SURFACE_VIEW_WIDTH;
    }

    public static int getSurfaceViewHeight() {
        return SURFACE_VIEW_HEIGHT;
    }

    private static int SURFACE_VIEW_WIDTH = 700;
    private static int SURFACE_VIEW_HEIGHT = 700;

    private static int PREVIEW_WIDTH = 0;
    private static int PREVIEW_HEIGHT = 0;

    public static int getScreenWidth() {
        return SCREEN_WIDTH;
    }

    public static int getScreenHeight() {
        return SCREEN_HEIGHT;
    }

    private static int SCREEN_WIDTH = 0;
    private static int SCREEN_HEIGHT= 0;

    public static Point getScreenSize(Activity context) {
        WindowManager windowManager = (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
        Display display = windowManager.getDefaultDisplay();
        Point outPoint = new Point();
        display.getSize(outPoint);
        if (isLandSpace(context)) {
            int y = outPoint.y;
            int x = outPoint.x;
            //outPoint.set(y,x);
        }
        LogUtil.i(TAG, "Get Screen Size" + "{" + " x:" + outPoint.x + ",y:" + outPoint.y + " }");
        SURFACE_VIEW_WIDTH = outPoint.x;
        SURFACE_VIEW_HEIGHT = outPoint.y;
        LogUtil.i(TAG, "Get Preview Size" + "{" + " x:" + SURFACE_VIEW_WIDTH + ",y:" + SURFACE_VIEW_HEIGHT + " }");
        return outPoint;
    }
    public static void getDisplaySize(Activity context) {
        DisplayMetrics dm = new DisplayMetrics();
        dm = context.getResources().getDisplayMetrics();
        SCREEN_WIDTH = dm.widthPixels;
        SCREEN_HEIGHT = dm.heightPixels;
        LogUtil.i(TAG, "Get Display Size" + "{" + " x:" + SCREEN_WIDTH + ",y:" + SCREEN_HEIGHT + " }");
    }



    public static Point getSurfacePoint(Activity activity) {
        Point windowSize = getScreenSize(activity);
        int centerY = windowSize.y / 2;
        int pointY = centerY - (SURFACE_VIEW_HEIGHT / 2);

        int three_twoX = windowSize.x / 3;
        int pointX = three_twoX - (SURFACE_VIEW_WIDTH / 2);
        Point outPoint = new Point();
        outPoint.set(pointX, pointY);
        LogUtil.i(TAG, "Get SurfaceView position" + "{" + " x:" + outPoint.x + ",y:" + outPoint.y + " }");
        return outPoint;
    }


    public static boolean isLandSpace(Context context) {
        int orientation = context.getResources().getConfiguration().orientation;
        if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
            LogUtil.i(TAG, "isLandSpace -->  " + true);
            return true;
        } else {
            LogUtil.i(TAG, "isLandSpace -->  " + false);
            return false;
        }
    }

    public static int px2dip(Context context, float px) {
        float density = context.getResources().getDisplayMetrics().density;
        return (int) (px * density + 0.5f);
    }

    public static int getPreviewWidth() {
        return PREVIEW_WIDTH;
    }

    public static int getPreviewHeight() {
        return PREVIEW_HEIGHT;
    }
}
