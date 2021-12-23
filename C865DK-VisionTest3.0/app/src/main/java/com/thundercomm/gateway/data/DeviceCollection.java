package com.thundercomm.gateway.data;

import android.annotation.SuppressLint;
import android.util.Log;

import com.thundercomm.eBox.Config.GlobalConfig;
import com.thundercomm.eBox.Constants.Constants;
import com.thundercomm.eBox.EventBusManager;
import com.thundercomm.eBox.Model.RtspItemCollection;
import com.thundercomm.eBox.Utils.LogUtil;
import com.thundercomm.eBox.Utils.PreferencesUtil;
import com.thundercomm.gateway.utils.JsonTools;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.thingsboard.server.common.data.kv.KvEntry;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.concurrent.CopyOnWriteArrayList;

import static com.thundercomm.eBox.Config.GlobalConfig.TESTDATA;
import static com.thundercomm.gateway.GateWayContanst.GATEWAY_CONF_DEVICE_ATTRIB;
import static com.thundercomm.gateway.GateWayContanst.GATEWAY_CONF_DEVICE_COUNT;
import static com.thundercomm.gateway.GateWayContanst.GATEWAY_CONF_DEVICE_KEY;
import static com.thundercomm.gateway.GateWayContanst.GATEWAY_DEVICE_ISREGIST;
import static com.thundercomm.gateway.GateWayContanst.GATEWAY_DEVICE_PATH;

public class DeviceCollection implements Serializable {
    private final String TAG = this.getClass().getName();
    private static final Object syncObj = new Object();
    private static DeviceCollection instance;
    private static CopyOnWriteArrayList<DeviceData> netDeviceList;
    private static CopyOnWriteArrayList<DeviceData> manuelDeviceList;
    private ArrayList<String> manuelRtspList;
    private ArrayList<String> netRtspList;
    private HashMap<Integer, String> attributeMap = new HashMap<>();
    private File file;
    private boolean readFromLocation = true;

    public static DeviceCollection getInstance() {
        if (instance == null) {
            synchronized (syncObj) {
                instance = new DeviceCollection();
            }
        }
        return instance;
    }

    public DeviceCollection() {
        init();
        EventBusManager.register(this);
    }

    @SuppressLint("AuthLeak")
    public void init() {
        initIOFile();
        initAttributeMap();
        initRtspList();
        readDeviceCollection();
    }

    private void initRtspList() {
        manuelRtspList = new ArrayList<>();
        netRtspList = new ArrayList<>();
        //face  -----------》
        //netRtspList.add("rtsp://admin:ZKCD1234@10.0.20.155:554/client0x");
        //netRtspList.add("rtsp://admin:ZKCD1234@10.0.20.155:554/client1x");
        netRtspList.add("rtsp://admin:ThunderSoft@172.18.0.156:554/client0x");
        //netRtspList.add("rtsp://admin:ThunderSoft@192.168.1.197:554/client1x");
        //netRtspList.add("rtsp://admin:ZKCD1234@10.0.20.155:554/client1x");

    }


    private void initIOFile() {
        try {
            file = new File(GlobalConfig.APP_FILE_PATH + GlobalConfig.APP_SERILIZEDATA_NAME);
            if (!file.exists()) {
                file.createNewFile();
                LogUtil.d(TAG, "initIOFile " + "createNewFile" + file.createNewFile());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void add(DeviceData deviceData) {
        Log.d(TAG, deviceData.toString());
        netDeviceList.add(deviceData);
    }

    public void remove(DeviceData deviceData) {
        Log.d(TAG, deviceData.toString());
        netDeviceList.remove(deviceData);
    }

    public void removeByName(String deviceName) {
        int i = -1;
        for (DeviceData deviceData : netDeviceList) {
            i++;
            if (deviceName != null && deviceName.contains(deviceData.getName())) {
                netDeviceList.remove(i);
            }
        }
    }

    public CopyOnWriteArrayList<DeviceData> getDeviceList() {
        if (PreferencesUtil.getInstance(null).getPreferenceString("get_url_type").equals("by_Manual")) {
            return manuelDeviceList;
        } else {
            return netDeviceList;
        }
    }

    public int getEnableVideoNum() {
        int num = 0;
        for (DeviceData deviceData : netDeviceList) {
            LogUtil.d(TAG, "getEnableVideoNum : ", getAttributesValue(deviceData, "enable_video"));
            if (Constants.SRT_TRUE.equals(getAttributesValue(deviceData, "enable_video"))) {
                num++;
            }
        }
        return num;
    }


    public void writeDeviceCollection() {
        try {
            ObjectOutputStream objectOutputStream;
            if (file == null) {
                initIOFile();
            } else {
                objectOutputStream = new ObjectOutputStream(new FileOutputStream(file));
                synchronized (syncObj) {
                    objectOutputStream.writeObject(getDeviceList());
                    LogUtil.v(TAG, "writeDeviceCollection--->" + GlobalConfig.APP_SERILIZEDATA_NAME);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public boolean deviceInList(String deviceName) {
        if (deviceName == null || deviceName.equals("")) {
            return false;
        }
        boolean res = false;
        Iterator iterator = this.getDeviceList().iterator();
        while (iterator.hasNext()) {
            DeviceData deviceData = (DeviceData) iterator.next();
            if (deviceData.getName().equals(deviceName)) {
                return true;
            }
        }
        return res;
    }


    private CopyOnWriteArrayList<DeviceData> readDeviceCollection() {
        try {
            ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream(file));
            if (file.exists()) {
                manuelDeviceList = (CopyOnWriteArrayList<DeviceData>) objectInputStream.readObject();
            } else {
                manuelDeviceList = new CopyOnWriteArrayList<>();
            }
        } catch (IOException | ClassNotFoundException e) {
            manuelDeviceList = new CopyOnWriteArrayList<DeviceData>();
        }

        if (TESTDATA) {
            initRtspList();
            netDeviceList = new CopyOnWriteArrayList<>();
            int num = netRtspList.size();
            for (int i = 0; i < num; i++) {
                String face_enable = "false";
                String car_enable = "false";
                String object_enable = "false";
                String flame_enable = "false";
                if (i < 1) {
                    face_enable = "true";
                } else {
                    object_enable = "true";
                }
                netDeviceList.add(DeviceData.builder().attributes(JsonTools.getKvEntries(
                        JsonTools.newNode().
                                put("URL", netRtspList.get(i)).
                                put("location", "位置_" + i).
                                put("enable_car", car_enable))).
                        name("Camera_" + i).build());
            }
        } else {
            netDeviceList = new CopyOnWriteArrayList<>();
            if (readFromLocation) {
                if (PreferencesUtil.getInstance(null).getPreferenceString(GATEWAY_DEVICE_ISREGIST).equals("true")) {
                    if (file.exists()) {
                        try {
                            ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream(file));
                            netDeviceList = (CopyOnWriteArrayList<DeviceData>) objectInputStream.readObject();
                        } catch (IOException e) {
                            e.printStackTrace();
                        } catch (ClassNotFoundException e) {
                            e.printStackTrace();
                        }
                        if (netDeviceList != null) {
                            Log.d(TAG, "GET Config From Serializable@Class " + netDeviceList.size());
                        }
                    }
                    if (netDeviceList == null) {
                        netDeviceList = new CopyOnWriteArrayList<>();
                    }
                }
            }
        }
        return netDeviceList;
    }


    private int getRtspNum() throws IOException, JSONException {
        String path = GATEWAY_DEVICE_PATH;
        StringBuffer stringBuffer = new StringBuffer();
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
        String line = "";
        while (true) {
            if (!((line = bufferedReader.readLine()) != null)) break;
            stringBuffer.append(line);
        }
        JSONObject gateWay_config = new JSONObject(stringBuffer.toString());
        return gateWay_config.getInt(GATEWAY_CONF_DEVICE_COUNT);
    }

    private DeviceData updateShareAttributes(String deviceName, String key, String value) {
        return updateShareAttributes(findDeviceByName(deviceName), key, value);
    }



    private DeviceData updateShareAttributes(int index, String key, String value) {
        DeviceData oldDeviceData = netDeviceList.get(index);
        synchronized (syncObj) {
            if (!"".equals(key)) {
                int i = -1;
                for (KvEntry kvEntry : oldDeviceData.getAttributes()) {
                    i++;
                    if (kvEntry.getKey().equals(key)) {
                        break;
                    }
                }
                if (i != -1) {
                    oldDeviceData.getAttributes().remove(i);
                    synchronized (netDeviceList) {
                        if (TESTDATA && key.equals("URL")) {
                            oldDeviceData.getAttributes().addAll(
                                    JsonTools.getKvEntries(JsonTools.newNode().put(key, netRtspList.get(index))));
                        } else {
                            oldDeviceData.getAttributes().addAll(
                                    JsonTools.getKvEntries(JsonTools.newNode().put(key, value)));
                        }
                        if (TESTDATA && key.equals("enable_video"))
                            oldDeviceData.getAttributes().addAll(
                                    JsonTools.getKvEntries(JsonTools.newNode().put(key, "true")));
                    }
                }
                // Log.d(TAG, "updateShareAttributes deveiceData" + oldDeviceData);
            }
        }
        return oldDeviceData;
    }


    public int findDeviceByName(String deviceName) {
        int index = -1;
        if (deviceName != null && !"".equals(deviceName)) {
            for (int i = 0, length = netDeviceList.size(); i < length; i++) {
                if (deviceName.equals(netDeviceList.get(i).getName())) {
                    index = i;
                    break;
                }
            }
        }
        return index;
    }

    public DeviceData findDeviceById(int index) {
        return netDeviceList.get(index);
    }


    public String getAttributesValue(DeviceData deviceData, String Key) {
        String res = "null";
        synchronized (syncObj) {
            if (deviceData != null) {
                for (KvEntry kvEntry : deviceData.getAttributes()) {
                    //Log.d(TAG, "Key " + kvEntry.getKey() + " value " + kvEntry.getValue());
                    if (kvEntry.getKey() != null) {
                        if (kvEntry.getKey().equals(Key)) {
                            res = kvEntry.getValueAsString();
                            break;
                        }
                    }
                }
            }
        }
        return res;
    }


    /**
     * 判断指定id的{@link DeviceData} 是否开启了该算法
     *
     * @param id      Fragment的Id
     * @param typeKey 关键字
     * @Describe
     */
    public static boolean getAlgorithmType(int id, String typeKey) {
        if (id > RtspItemCollection.getInstance().getDeviceList().size()) {
            return false;
        }
        DeviceData deviceData = RtspItemCollection.getInstance().getDeviceList().get(id);
        boolean enable = Boolean.parseBoolean(RtspItemCollection.getInstance().getAttributesValue(deviceData, typeKey));
        LogUtil.d("TAG-getAlgorithmType", "getAlgorithmType index->" + id + "  type:" + typeKey + "result:" + enable);
        return enable;
    }


    private void initAttributeMap() {
        attributeMap = new HashMap<>();
        try {
            String path = GATEWAY_DEVICE_PATH;
            StringBuffer stringBuffer = new StringBuffer();
            BufferedReader bufferedReader = null;
            bufferedReader = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line = "";
            while (true) {
                try {
                    if (!((line = bufferedReader.readLine()) != null)) break;
                } catch (IOException e) {
                    e.printStackTrace();
                }
                stringBuffer.append(line);
            }
            JSONObject gateWay_config = new JSONObject(stringBuffer.toString());
            //get attributes config from json
            JSONArray device_attributes = gateWay_config.getJSONArray(GATEWAY_CONF_DEVICE_ATTRIB);
            for (int i = 0; i < device_attributes.length(); i++) {
                JSONObject attributes_Node = (JSONObject) device_attributes.get(i);
                String key = attributes_Node.getString(GATEWAY_CONF_DEVICE_KEY);
                attributeMap.put(i, key);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (JSONException e) {
            e.printStackTrace();
        }
        LogUtil.i(TAG, "Init Attribute Map " + attributeMap);
    }
}
