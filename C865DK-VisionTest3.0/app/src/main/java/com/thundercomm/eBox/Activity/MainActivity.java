package com.thundercomm.eBox.Activity;

import android.annotation.SuppressLint;
import android.app.ActivityManager;
import android.app.AlertDialog;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentTransaction;

import com.thundercomm.eBox.AI.GestureClassificationTask;
import com.thundercomm.eBox.AI.OPencvInit;
import com.thundercomm.eBox.Config.GlobalConfig;
import com.thundercomm.eBox.Constants.Constants;
import com.thundercomm.eBox.Model.RtspItemCollection;
import com.thundercomm.eBox.R;
import com.thundercomm.eBox.Utils.ActivityUtil;
import com.thundercomm.eBox.Utils.LogUtil;
import com.thundercomm.eBox.Utils.PreferencesUtil;
import com.thundercomm.eBox.Utils.WindowUtils;
import com.thundercomm.eBox.VIew.GestureClassificationFragment;
import com.thundercomm.eBox.VIew.PlayFragment;
import com.thundercomm.gateway.data.DeviceCollection;

import java.util.ArrayList;
import java.util.TimerTask;
import java.util.Vector;

import butterknife.ButterKnife;

import static android.view.MotionEvent.ACTION_DOWN;
import static com.thundercomm.eBox.Config.GlobalConfig.SAVE_PATH;
import static com.thundercomm.eBox.Config.GlobalConfig.VIDEO_MARGIN;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

/**
 * MainActivity 用于视频解码展示和算法结果展示

 */
public class MainActivity extends BaseActivity implements View.OnClickListener {
    private final static int STATUS_UNPREPARE = 0;
    private final static int STATUS_PREPARE_S1 = 1;
    private final static int STATUS_PREPARE_S2 = 2;
    private final long bootTime = System.currentTimeMillis();
    private String TAG = "MainActivity";
    private ArrayList<LinearLayout> linearLayoutArrayList = new ArrayList<>();
    private ArrayList<PlayFragment> fragmentManageList = new ArrayList<>();
    private ArrayList<PlayFragment.MyOnTouchListener> listenerArrayList = new ArrayList<>();
    private LinearLayout parentLinearLayout;
    private Vector<Integer> idList;
    private TextView tv_title;

    private int row = 1;
    private int col = 1;
    private int rtspNum = 0;

    private boolean isAllFragmentPrepared = false;

    private ImageButton settingIButton;
    private ImageButton startIButton;
    private ImageButton logoutIButton;

    private LinearLayout constraintLayout;
    @SuppressLint("ResourceType")
    private int layoutResId = 1001;

    private int isAllowClick = STATUS_UNPREPARE;
    private long startTime;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    LogUtil.i(TAG, "OpenCV loaded successfully");
                    // TODO: enable opencv view
                    OPencvInit.setLoaderOpenCV(true);
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };
    private Handler handler;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        LogUtil.e(TAG, "onCreate");
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ButterKnife.bind(this);
        initResourceFile();
        initData();
        startTime = System.currentTimeMillis();
    }

    @Override
    public boolean dispatchTouchEvent(MotionEvent ev) {
        if (ev.getAction() == ACTION_DOWN) {
            int index = getTouchViewIndex(ev);
            LogUtil.d(TAG, "getTouchViewIndex --->" + index);
            //Toast.makeText(getBaseContext(), "You touch fragment --->" + index, Toast.LENGTH_SHORT).show();
            if (listenerArrayList.size() > 0 && index >= 0) {
                listenerArrayList.get(index).onTouch(ev, index);
            }
        }
        return super.dispatchTouchEvent(ev);
    }

    /**
     * 初始化资源文件 调用{@link MainActivity#moveAssetToPath(String, String)}将Asset的文件移动到指定目录下
     *
     * @Describe 初始化资源文件
     */
    private void initResourceFile() {
        moveAssetToPath("edgebox_gatewayConfig.txt", SAVE_PATH);
        moveAssetToPath("edgebox_gatewayConfig_template.txt", SAVE_PATH);
        moveAssetToPath("edgebox_smartRetailConfig.txt", SAVE_PATH);
    }


    @Override
    protected void onResume() {
        isAllowClick = STATUS_UNPREPARE;
        super.onResume();
        initModfiyLinearLayout();
        initView();
        LogUtil.e(TAG, "onResume");
        //添加fragment
        handler = new Handler();
        fragmentManageList.clear();
        for (int i = 0; i < rtspNum; i++) {
            final int final_I = i;
            PlayFragment playFragment = createPlayFragment(final_I);
            playFragment.setMyOnTouchListener(myOnTouchListener);
            listenerArrayList.add(myOnTouchListener);
            fragmentManageList.add(playFragment);
            idList.add(final_I);
            handler.postDelayed(() -> addNewFragment(linearLayoutArrayList.get(final_I), playFragment), play_stream_counts * 500);
            play_stream_counts++;
        }
        handler.obtainMessage();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

    }

    private void startAIThread() {
        AIStartThread startThread = new AIStartThread();
        startThread.start();
    }

    @Override
    protected void onPause() {
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        stopAIService();
        isAllFragmentPrepared = false;
        ActivityUtil.getInstance().finishAllActivity();
        super.onDestroy();

    }

    /**
     * 生成自定义的动态布局
     */
    @SuppressLint("ResourceType")
    private void initModfiyLinearLayout() {
        Runtime.getRuntime().gc();
        rtspNum = RtspItemCollection.getInstance().getDeviceList().size();
        if (rtspNum == 0) {
            isAllowClick++;
        }
        parentLinearLayout = new LinearLayout(this); // 生成父布局
        if (rtspNum == 1) {
            row = 1;
            col = 1;
        } else if (rtspNum <= 4) {
            row = (int) Math.sqrt(4);
            col = (int) Math.sqrt(4);
        } else if (rtspNum <= 9) {
            row = (int) Math.sqrt(9);
            col = (int) Math.sqrt(9);
        } else if (rtspNum <= 16) {
            row = (int) Math.sqrt(16);
            col = (int) Math.sqrt(16);
        } else if (rtspNum == 24) {
            row = 4;
            col = 6;
        } else if (rtspNum <= 25 && rtspNum != 24) {
            row = (int) Math.sqrt(25);
            col = (int) Math.sqrt(25);
        }
        // 使用linearLayout构成网格布局
        LinearLayout.LayoutParams layoutParams = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.MATCH_PARENT);
        parentLinearLayout.setId(layoutResId);
        parentLinearLayout.setLayoutParams(layoutParams);
        parentLinearLayout.setOrientation(LinearLayout.VERTICAL);
        constraintLayout = findViewById(R.id.main_VideoLayout);
        constraintLayout.addView(parentLinearLayout);
        for (int i = 0; i < row; i++) {
            LinearLayout rowChildLayout = new LinearLayout(this);
            LinearLayout.LayoutParams childLayoutParams = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1.0f);
            rowChildLayout.setOrientation(LinearLayout.HORIZONTAL);
            rowChildLayout.setId(++layoutResId);
            rowChildLayout.setLayoutParams(childLayoutParams);
            for (int j = 0; j < col; j++) {
                LinearLayout colChildLayout = new LinearLayout(this);
                LinearLayout.LayoutParams colChildLayoutParams = new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1.0f);
                colChildLayoutParams.bottomMargin = VIDEO_MARGIN;
                colChildLayoutParams.topMargin = VIDEO_MARGIN;
                colChildLayoutParams.leftMargin = VIDEO_MARGIN;
                colChildLayoutParams.rightMargin = VIDEO_MARGIN;
                colChildLayout.setWeightSum(1);
                colChildLayout.setId(++layoutResId);
                colChildLayout.setLayoutParams(colChildLayoutParams);
                rowChildLayout.addView(colChildLayout);
                linearLayoutArrayList.add(colChildLayout);
            }
            parentLinearLayout.addView(rowChildLayout);
        }

        PreferencesUtil.getInstance(this).setPreferences("ROWNUM", String.valueOf(rtspNum));
    }

    @Override
    protected void onStart() {
        super.onStart();
    }

    class AIStartThread extends Thread {
        @Override
        public void run() {
            super.run();
            try {
                Thread.sleep(1000 * 3 );
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            if (GlobalConfig.ENABLE_MUTILOBJECTDETECTION) {
                GestureClassificationTask.getGestureClassificationTask().init(getApplication(),
                        idList,
                        fragmentManageList,
                        GlobalConfig.INPUT_WIDTH,
                        GlobalConfig.INPUT_HEIGHT);
            }
            isAllowClick++;
        }
    }

    private static int play_stream_counts = 1;

    private void initView() {

        settingIButton = findViewById(R.id.ll_st_setting_ibtn);
        logoutIButton = findViewById(R.id.ibtn_logout);
        startIButton = findViewById(R.id.ll_start);

        logoutIButton.setOnClickListener(onClickListener);
        settingIButton.setOnClickListener(onClickListener);
        startIButton.setOnClickListener(onClickListener);
        final View view = getWindow().getDecorView();
        int uiOptions = View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY | View.SYSTEM_UI_FLAG_FULLSCREEN;
        view.setSystemUiVisibility(uiOptions);
        idList = new Vector<Integer>();

        isAllFragmentPrepared = true;

        //设置标题阴影
        tv_title = findViewById(R.id.ll_smartrretail_tv);
        tv_title.setShadowLayer(15, 0, 0, getResources().getColor(R.color.color_1da8e9));
    }

    /**
     * 添加新的Fragment
     *
     * @param linearLayout1 目标布局
     * @param playFragment  要添加的Fragment
     * @Describe 添加新的Fragment
     */
    private void addNewFragment(LinearLayout linearLayout1, PlayFragment playFragment) {
        FragmentManager fragmentManager = getSupportFragmentManager();
        FragmentTransaction fragmentTransaction = fragmentManager.beginTransaction();
        fragmentTransaction.add(linearLayout1.getId(), playFragment);
        fragmentTransaction.commit();
        if (linearLayoutArrayList.indexOf(linearLayout1) == (rtspNum - 1)) {
            isAllowClick++;
        }
    }

    private void replaceFragment(LinearLayout linearLayout, PlayFragment
            oldFragment, PlayFragment newFragment, int index) {
        if (linearLayout.getChildCount() >= 0) {
            LogUtil.e(TAG, fragmentManageList.toString());
            FragmentManager fragmentManager = getSupportFragmentManager();
            FragmentTransaction fragmentTransaction = fragmentManager.beginTransaction();
            fragmentTransaction.replace(oldFragment.getId(), newFragment);
            fragmentTransaction.commit();
            fragmentManageList.remove(index);
            fragmentManageList.add(index, newFragment);
        }
    }

    private void removeAllFragment() {
        FragmentManager fragmentManager = getSupportFragmentManager();
        FragmentTransaction fragmentTransaction = fragmentManager.beginTransaction();
        for (PlayFragment playFragment : fragmentManageList) {
            fragmentTransaction.remove(playFragment);
        }
        fragmentTransaction.commit();
        for (LinearLayout linearLayout1 : linearLayoutArrayList) {
            linearLayout1.removeAllViews();
        }
        linearLayoutArrayList.clear();
        constraintLayout.removeAllViews();

        isAllFragmentPrepared = false;

        System.gc();
        System.gc();
    }

    private void initData() {
        WindowUtils.getScreenSize(this);
        WindowUtils.getDisplaySize(this);
        PreferencesUtil.getInstance(getBaseContext());
    }

    @Override
    public void onBackPressed() {
        for (PlayFragment fragment : fragmentManageList) {
            fragment.onDestroy();
        }
        super.onBackPressed();
    }

    @Override
    public void onClick(View v) {
        LogUtil.d(TAG, "Click --->" + v.getId());
        switch (v.getId()) {
        }
    }

    private void shoutDown() {
        for (PlayFragment playFragment : fragmentManageList) {
            playFragment.onDestroy();
        }
        for (LinearLayout linearLayout : linearLayoutArrayList) {
            linearLayout.removeAllViews();
        }
    }

    private class ReMemConnectTimerTask extends TimerTask {

        @Override
        public void run() {
            ActivityManager activityManager = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
            //最大分配内存
            int memory = activityManager.getMemoryClass();
            Log.d("mem", "memory: " + memory);
            //最大分配内存获取方法2
            float maxMemory = (float) (Runtime.getRuntime().maxMemory() * 1.0 / (1024 * 1024));
            //当前分配的总内存
            float totalMemory = (float) (Runtime.getRuntime().totalMemory() * 1.0 / (1024 * 1024));
            //剩余内存
            float freeMemory = (float) (Runtime.getRuntime().freeMemory() * 1.0 / (1024 * 1024));
            Log.d("mem", "maxMemory: " + maxMemory);
            Log.d("mem", "totalMemory: " + totalMemory);
            Log.d("mem", "freeMemory: " + freeMemory);
        }
    }


    private long lastClickTime = System.currentTimeMillis();
    private View.OnClickListener onClickListener = v -> {
        if (isAllFragmentPrepared) {
            LogUtil.e(TAG, "Click!!!!!!!");
            switch (v.getId()) {
                case R.id.ibtn_logout:
                    break;
                case R.id.ibtn_logo:
//                    LogUtil.e(TAG, "Click!!!!!!!");
//                    stopAIService();
//                    removeAllFragment();
//                    startActivity(new Intent(MainActivity.this, SettingActivity.class));
                    break;
                case R.id.ll_st_setting_ibtn:
                    LogUtil.d(TAG, "isAllowClick-->", isAllowClick);
                    removeAllFragment();
                    try {
                        Thread.sleep(Constants.NUM_1000);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    try {
                        Thread.sleep(Constants.NUM_3000);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    startActivity(new SettingsActivity());

                    break;

                case R.id.ll_start:
                    AlertDialog.Builder builder = new AlertDialog.Builder(this);
                    builder.setTitle("The method of test vision:");
                    builder.setMessage(
                            "1. Approximately 1.5 meters away from the screen, viewing in parallel.\n"
                            + "2. The subject gently covers the left eye with hand, and at the same time, open both eyes.\n"
                            + "3. According to the \"E\" displayed on the upper left of the "
                            + "screen, use gestures to point out the direction.\n"
                            + "4. After the right eye is tested, the same procedure is used to test the left eye."
                    );
                    builder.setPositiveButton("start to test", (dialog, which) -> {
                        final View view = getWindow().getDecorView();
                        int uiOptions = View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                                | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY | View.SYSTEM_UI_FLAG_FULLSCREEN;
                        view.setSystemUiVisibility(uiOptions);
                        startAIThread();
                    });
                    builder.create().show();

                    break;
                default:
                    break;
            }
        } else {
            Toast.makeText(getBaseContext(), "ALL VIEW HAVE NOT PREPARED!", Toast.LENGTH_SHORT).show();
        }

    };

    /**
     * 停止开启的AI算法线程
     */
    private void stopAIService() {
        if (GlobalConfig.ENABLE_MUTILOBJECTDETECTION) {
            GestureClassificationTask.getGestureClassificationTask().closeService();
        }

        Runtime.getRuntime().gc();
    }

    private int getTouchViewIndex(MotionEvent ev) {
        int index = -1;
        for (PlayFragment playFragment : fragmentManageList) {
            index++;
            View view = playFragment.getView();
            if (view != null) {
                RectF rectF = calcViewScreenLocation(view);
                float x = ev.getRawX(); // 获取相对于屏幕左上角的 x 坐标值
                float y = ev.getRawY(); // 获取相对于屏幕左上角的 y 坐标值
                boolean isIn = rectF.contains(x, y);
                if (isIn) {
                    return index;
                }
            }
        }
        return -1;
    }

    /**
     * 计算指定的 View 在屏幕中的坐标。
     */
    public static RectF calcViewScreenLocation(View view) {
        int[] location = new int[2];
        // 获取控件在屏幕中的位置，返回的数组分别为控件左顶点的 x、y 的值
        view.getLocationOnScreen(location);
        return new RectF(location[0], location[1], location[0] + view.getWidth(),
                location[1] + view.getHeight());
    }

    PlayFragment.MyOnTouchListener myOnTouchListener = (event, index) -> {

    };

    /**
     * 重新加载一个playFragment
     *
     * @param index 在fragmentManageList中的位置
     * @Describe 重新加载一个playFragment
     */
    private void reloadFragment(int index) {
        fragmentManageList.get(index).onDestroy();
        PlayFragment playFragment = createPlayFragment(index);
        playFragment.setMyOnTouchListener(myOnTouchListener);
        replaceFragment(linearLayoutArrayList.get(index), fragmentManageList.get(index), playFragment, index);
        replaceListener(index, myOnTouchListener);
        for (PlayFragment playFragment1 : fragmentManageList) {
            LogUtil.d(TAG, "PlayFragMents" + playFragment1.getId());
        }
    }

    public void replaceListener(int index, PlayFragment.MyOnTouchListener myOnTouchListener) {
        listenerArrayList.remove(index);
        listenerArrayList.add(index, myOnTouchListener);
    }

    /**
     * 通过id判断生成那种碎片
     *
     * @param index id
     * @Describe 判断生成哪种碎片
     */
    private PlayFragment createPlayFragment(int index) {
        if (DeviceCollection.getAlgorithmType(index, Constants.ENABLE_GESTURECLASSIFICATION_STR)) {
            return new GestureClassificationFragment(index);
        }
        return new PlayFragment(index);
    }

}
