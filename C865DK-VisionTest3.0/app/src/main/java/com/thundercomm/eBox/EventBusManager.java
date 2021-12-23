package com.thundercomm.eBox;

import android.graphics.Rect;




import org.greenrobot.eventbus.EventBus;

import java.util.ArrayList;
import java.util.Vector;


public class EventBusManager {
    //开启Index加速
    public static void openIndex() {
        //  EventBus.builder().addIndex(new MyEventBusIndex()).installDefaultEventBus();
    }

    //订阅事件
    public static void register(Object subscriber) {
        if (!EventBus.getDefault().isRegistered(subscriber)) {
            //   EventBus.getDefault().register(subscriber);
        }
    }

    //取消订阅
    public static void unregister(Object subscriber) {
//        if (EventBus.getDefault().isRegistered(subscriber)) {
//            EventBus.getDefault().unregister(subscriber);
//        }
    }

    //终止事件继续传递
    public static void cancelDelivery(Object event) {
        EventBus.getDefault().cancelEventDelivery(event);
    }

    //获取保存起来的粘性事件
    public static <T> T getStickyEvent(Class<T> classType) {
        return EventBus.getDefault().getStickyEvent(classType);
    }

    //删除保存中的粘性事件
    public static void removeStickyEvent(Object event) {
        EventBus.getDefault().removeStickyEvent(event);
    }

    //发送事件
    public static void postEvent(Object event) {
        EventBus.getDefault().post(event);
    }

    //发送粘性事件
    public static void postStickyEvent(Object event) {
        EventBus.getDefault().postSticky(event);
    }



}
