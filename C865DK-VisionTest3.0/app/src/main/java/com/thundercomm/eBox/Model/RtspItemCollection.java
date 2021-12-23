package com.thundercomm.eBox.Model;

import androidx.annotation.Nullable;

import com.thundercomm.eBox.EventBusManager;
import com.thundercomm.gateway.data.DeviceCollection;
import com.thundercomm.gateway.data.DeviceData;

import java.io.Serializable;
import java.util.ArrayList;


public class RtspItemCollection extends DeviceCollection {
    private String TAG = "RtspItemCollection";

    public RtspItemCollection() {
        init();
        EventBusManager.register(this);
    }

    private static ArrayList<RtspItem> rtspItems = new ArrayList<>();

    public static ArrayList<RtspItem> getRtspItems() {
        return rtspItems;
    }

    public static void setRtspItems(ArrayList<RtspItem> rtspItems) {
        RtspItemCollection.rtspItems = rtspItems;
    }

    public static int getSize() {
        return rtspItems.size();
    }

    public static void add(RtspItem item) {
        rtspItems.add(item);
    }

    public static void removeAll() {
        rtspItems.clear();
    }

    public DeviceData get(int i) {
        return getDeviceList().get(i);
    }

    public static class RtspItem implements Serializable {

        private String url;
        private String width = "1920";
        private String height = "1080";
        private String decodeType = "H264";
        private String frameRate = "25";
        private String enable_video;
        private String enable_car;
        private String enable_face;
        private String location = "";

        public String getUrl() {
            return url;
        }

        public void setUrl(String url) {
            this.url = url;
        }

        public String getWidth() {
            return width;
        }

        public void setWidth(@Nullable String width) {
            this.width = width;
        }

        public String getHeight() {
            return height;
        }

        public void setHeight(@Nullable String height) {
            this.height = height;
        }

        public String getDecodeType() {
            return decodeType;
        }

        public void setDecodeType(@Nullable String decodeType) {
            this.decodeType = decodeType;
        }

        public String getFrameRate() {
            return frameRate;
        }

        public void setFrameRate(@Nullable String frameRate) {
            this.frameRate = frameRate;
        }

        public String getEnable_video() {
            return enable_video;
        }

        public void setEnable_video(@Nullable String enable_video) {
            this.enable_video = enable_video;
        }

        public String getEnable_car() {
            return enable_car;
        }

        public void setEnable_car(@Nullable String enable_car) {
            this.enable_car = enable_car;
        }

        public String getEnable_face() {
            return enable_face;
        }

        public void setEnable_face(@Nullable String enable_face) {
            this.enable_face = enable_face;
        }

        public String getLocation() {
            return location;
        }

        public void setLocation(@Nullable String location) {
            this.location = location;
        }

        public RtspItem(String url, String width, String height, String decodeType, String frameRate, String enable_video, String enable_car, String enable_face, String location) {
            this.url = url;
            this.width = width;
            this.height = height;
            this.decodeType = decodeType;
            this.frameRate = frameRate;
            this.enable_video = enable_video;
            this.enable_car = enable_car;
            this.enable_face = enable_face;
            this.location = location;
        }

        public RtspItem() {

        }

        @Override
        public String toString() {
            return "RtspItem{" +
                    "url='" + url + '\'' +
                    ", location='" + location + '\'' +
                    ", width='" + width + '\'' +
                    ", height='" + height + '\'' +
                    ", decodeType='" + decodeType + '\'' +
                    ", frameRate='" + frameRate + '\'' +
                    ", enable_video='" + enable_video + '\'' +
                    ", enable_car='" + enable_car + '\'' +
                    ", enable_face='" + enable_face + '\'' +

                    '}';
        }
    }
}
