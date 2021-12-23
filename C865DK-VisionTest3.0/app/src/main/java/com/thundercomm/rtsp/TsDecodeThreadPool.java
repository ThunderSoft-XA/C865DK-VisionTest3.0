package com.thundercomm.rtsp;

import com.fasterxml.jackson.core.TSFBuilder;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class TsDecodeThreadPool {
    private static TsDecodeThreadPool instance;
    private static final Object syncObj = new Object();

    public ExecutorService getDecodeThreadPool() {
        return decodeThreadPool;
    }

    private ExecutorService decodeThreadPool;

    public static TsDecodeThreadPool getInstance() {
        if (instance == null) {
            synchronized (syncObj) {
                instance = new TsDecodeThreadPool();

            }
        }
        return instance;
    }

    private TsDecodeThreadPool() {
        decodeThreadPool = Executors.newCachedThreadPool();
    }

}
