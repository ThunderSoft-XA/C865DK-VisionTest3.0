package com.thundercomm.eBox.Activity;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlarmManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.FragmentActivity;

import com.thundercomm.eBox.Config.GlobalConfig;
import com.thundercomm.eBox.Constants.Constants;

import com.thundercomm.eBox.Utils.ActivityUtil;
import com.thundercomm.eBox.Utils.FileUtil;
import com.thundercomm.eBox.Utils.LogUtil;
import com.thundercomm.eBox.Utils.NetWorkSpeedUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * 基础Activity类 实现权限申请，网络监听，标题栏隐藏等
 */
@SuppressLint("Registered")
public class BaseActivity extends FragmentActivity {
    private String TAG = this.getClass().getName();

    /**
     * android storage permission
     */
    private static final int PERMISSIONS_REQUEST = 1;
    private static final String PERMISSION_STORAGE = Manifest.permission.WRITE_EXTERNAL_STORAGE;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        LogUtil.v(TAG, "--OnCreate ");
        this.requestWindowFeature(Window.FEATURE_NO_TITLE);
        super.onCreate(savedInstanceState);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        if (!hasPermission()) {
            requestPermission();
        }
        initNetWorkManager();
        setSaveDir();
        FileUtil.getInstance(this);

        ActivityUtil.getInstance().addActivity(this);
    }

    @Override
    protected void onResume() {
        hideBottomUIMenu();
        //设置横屏
        if (getRequestedOrientation() != ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE) {
            setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        }
        super.onResume();
    }

    protected void hideBottomUIMenu() {
        //隐藏虚拟按键，并且全屏
        if (Build.VERSION.SDK_INT > 11 && Build.VERSION.SDK_INT < 19) { // lower api
            View v = this.getWindow().getDecorView();
            v.setSystemUiVisibility(View.GONE);
        } else if (Build.VERSION.SDK_INT >= 19) {
            //for new api versions.
            View decorView = getWindow().getDecorView();
            int uiOptions = View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                    | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY | View.SYSTEM_UI_FLAG_FULLSCREEN;
            decorView.setSystemUiVisibility(uiOptions);
        }
    }

    /**
     * 初始化网络监听
     * @Describe 初始化网络监听
     */
    private void initNetWorkManager() {
        NetWorkSpeedUtils netWorkSpeedUtils = new NetWorkSpeedUtils(BaseActivity.this, new Handler());
        netWorkSpeedUtils.startShowNetSpeed();
    }

    /**
     * 初始化app路径
     * @Describe 初始化app路径
     */
    private void setSaveDir() {
        File file = new File(GlobalConfig.SAVE_PATH);
        if (!file.exists()) {
            file.mkdir();
        }
    }


    @Override
    protected void onDestroy() {
        ActivityUtil.getInstance().removeActivity(this);
        super.onDestroy();
    }

    private static String[] permissionStr = {
            Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.INTERNET
    };

    protected void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_STORAGE)) {
                Toast.makeText(BaseActivity.this,
                        "Storage permission are required for this demo", Toast.LENGTH_LONG).show();
            }
            requestPermissions(permissionStr, PERMISSIONS_REQUEST);
        }
    }

    protected boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(PERMISSION_STORAGE) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        boolean isGranted = false;
        if (requestCode == PERMISSIONS_REQUEST) {
            if (grantResults.length >= 1 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                isGranted = true;
            }
        }
        if (isGranted) {

        } else {
            finish();
        }
    }

    /**
     * 重启app
     * @Describe 重启app
     */
    void rebootApp() {
        Intent intent = getBaseContext().getPackageManager().getLaunchIntentForPackage(getBaseContext().getPackageName());
        PendingIntent restartIntent = PendingIntent.getActivity(getApplicationContext(), 0, intent, PendingIntent.FLAG_ONE_SHOT);
        AlarmManager mgr = (AlarmManager) getSystemService(Context.ALARM_SERVICE);
        mgr.set(AlarmManager.RTC, System.currentTimeMillis() + 1000, restartIntent); // 1秒钟后重启应用
        onDestroy();
    }

    /**
     * 移动Asset的文件到指定路径下
     *
     * @param srcFilePath 原路径 可以是文件或文件夹
     * @param targetPath  目标路径
     * @Describe 移动Asset的文件到指定路径下
     */
    public void moveAssetToPath(String srcFilePath, String targetPath) {
        Context context = getBaseContext();
        try {
            String fileNames[] = context.getAssets().list(srcFilePath);//获取assets目录下的所有文件及目录名
            if (fileNames.length > 0) {//如果是目录 批量移动
                File file = new File(targetPath);
                file.mkdirs();//如果文件夹不存在，则递归
                for (String fileName : fileNames) {
                    copyFilesFassets(srcFilePath, fileName, targetPath);
                }
            } else {//如果是文件
                InputStream is = getBaseContext().getAssets().open(srcFilePath);
                FileOutputStream fos = new FileOutputStream(new File(targetPath + "/" + srcFilePath));
                byte[] buffer = new byte[1024];
                int byteCount = 0;
                while ((byteCount = is.read(buffer)) != -1) {//循环从输入流读取 buffer字节
                    fos.write(buffer, 0, byteCount);//将读取的输入流写入到输出流
                }
                fos.flush();//刷新缓冲区
                is.close();
                fos.close();
            }
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    private void copyFilesFassets(String srcFilePath, String fileName, String targetPath) {
        try {
            InputStream is = getBaseContext().getAssets().open(srcFilePath + "/" + fileName);
            File targetDir = new File(targetPath + "/" + srcFilePath);
            if (!targetDir.exists()) {
                targetDir.mkdir();
            }
            FileOutputStream fos = new FileOutputStream(new File(targetPath + "/" + srcFilePath + "/" + fileName));
            byte[] buffer = new byte[1024];
            int byteCount = 0;
            while ((byteCount = is.read(buffer)) != -1) {//循环从输入流读取 buffer字节
                fos.write(buffer, 0, byteCount);//将读取的输入流写入到输出流
            }
            fos.flush();//刷新缓冲区
            is.close();
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void startActivity(Activity activity) {
        Intent i = new Intent(this, activity.getClass());
        startActivity(i);
    }
}
