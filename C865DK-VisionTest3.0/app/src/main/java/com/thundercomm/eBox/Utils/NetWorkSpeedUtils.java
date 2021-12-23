package com.thundercomm.eBox.Utils;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.net.TrafficStats;
import android.os.Handler;

import java.util.Timer;
import java.util.TimerTask;



public class NetWorkSpeedUtils {
    private String TAG = "NetWorkSpeedUtils";
    private Context context;
    private Handler mHandler;
    private Timer speedTimer;
    private Timer netStatusTimer;
    private long warningSpeed;

    private long lastTotalTxBytes = 0;
    private long lastTimeStamp = 0;

    public NetWorkSpeedUtils(Context context, Handler mHandler) {
        this.context = context;
        this.mHandler = mHandler;
        speedTimer = new Timer();
        netStatusTimer = new Timer();
    }

    TimerTask task = new TimerTask() {
        @Override
        public void run() {
            showNetSpeed();
        }
    };

    TimerTask netStatus = new TimerTask() {
        @Override
        public void run() {
            checkNetStatus();
        }
    };

    private void checkNetStatus() {
        if (context != null) {
            ConnectivityManager connectivityManager = (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
            NetworkInfo networkInfo = connectivityManager.getActiveNetworkInfo();

        }
    }

    public void startShowNetSpeed() {
        lastTotalTxBytes = getTotalRxBytes();
        lastTimeStamp = System.currentTimeMillis();
        speedTimer.schedule(task, 1000, 1000); // 1s后启动任务，每2s执行一次
        netStatusTimer.schedule(netStatus, 1000, 5000);
    }

    public void stopShowNetSpeed() {
        speedTimer.cancel();
        netStatusTimer.cancel();
    }

    private long getTotalRxBytes() {
        return TrafficStats.getUidRxBytes(context.getApplicationInfo().uid) == TrafficStats.UNSUPPORTED ? 0 : (TrafficStats.getTotalRxBytes() / 1024);//转为KB
    }

    private void showNetSpeed() {
        long totalTxBytes = getTotalRxBytes();
        long nowTimeStamp = System.currentTimeMillis();
        long speed = ((totalTxBytes - lastTotalTxBytes) * 1000 / (nowTimeStamp - lastTimeStamp));//毫秒转换
        long speed2 = ((totalTxBytes - lastTotalTxBytes) * 1000 % (nowTimeStamp - lastTimeStamp));//毫秒转换

        lastTimeStamp = nowTimeStamp;
        lastTotalTxBytes = totalTxBytes;
        LogUtil.d(TAG, "Receive Speed --->", speed + "kb/s");
        if (speed < warningSpeed) {
            LogUtil.e(TAG, "Receive Speed ---> Too low speed(", speed + "kb/s)");
        }
    }

}