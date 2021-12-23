package com.thundercomm.eBox.Config;

import android.os.Environment;

import java.io.File;

public class GlobalConfig {
    public static boolean IFPREVIEW = true;


    public static String SAVE_PATH = Environment.getExternalStorageDirectory().getPath() + File.separator + "EDGEBOX/";
    public static String APP_FILE_PATH = SAVE_PATH;
    public static String APP_SERILIZEDATA_NAME = "devicelist.txt";
    public static boolean CONNECT_TO_WEB = true;                  //是否连接至web端
    public static boolean TESTDATA = true;                         //是否使用测试数据
    public static int VIDEO_MARGIN = 5; // 视频view之间的间隔
    public static int MAX_CAMERA_NUM = 24; //最大可用Camera数目

    public static final boolean ENABLE_MUTILOBJECTDETECTION = true;
    public static final int INPUT_WIDTH = 224;
    public static final int INPUT_HEIGHT = 224;
}
