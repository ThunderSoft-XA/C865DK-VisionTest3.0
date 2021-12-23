package com.thundercomm.gateway;

import android.os.Environment;

import java.io.File;

/**

 * @Describe
 */
public class GateWayContanst {
    public static String GATEWAY_CONF_DEVICE_COUNT = "device_count";
    public static String GATEWAY_CONF_DEVICE_TYPE = "device_type";
    public static String GATEWAY_CONF_DEVICE_ATTRIB = "device_attributes";
    public static String GATEWAY_CONF_DEVICE_TELEM = "devices_telemetry";

    public static String GATEWAY_CONF_DEVICE_KEY = "key";
    public static String GATEWAY_CONF_DEVICE_VALUE = "value";

    public static String GATEWAY_DEVICE_NODE_ID ="id";
    public static String GATEWAY_DEVICE_NODE_DNAME = "device";
    public static String GATEWAY_DEVICE_NODE_DATA = "data";

    public static String GATEWAY_DEVICE_ISREGIST = "ISREGISTDEVICE";

    public static String GATEWAY_DEVICE_PATH = Environment.getExternalStorageDirectory().getPath() + File.separator + "EDGEBOX/edgebox_smartRetailConfig.txt";
    public static String GATEWAY_DEVICE_PATH_back= Environment.getExternalStorageDirectory().getPath() + File.separator + "EDGEBOX/edgebox_gatewayConfig.txt";
    public static String GATEWAY_SMARTRETAIL_PATH = Environment.getExternalStorageDirectory().getPath() + File.separator + "EDGEBOX/edgebox_smartRetailConfig.txt";
    public static String GATEWAY_DEVICESERIA_PATH = Environment.getExternalStorageDirectory().getPath() + File.separator + "EDGEBOX/DeviceCollection.txt";
}
