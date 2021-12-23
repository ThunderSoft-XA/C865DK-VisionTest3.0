package com.thundercomm.eBox.Log;

import com.orhanobut.logger.AndroidLogAdapter;
import com.orhanobut.logger.Logger;

public class LogManager {
    private boolean debuglog = false;

    public LogManager(boolean enablelog) {
        this.debuglog = enablelog;

        Logger.addLogAdapter(new AndroidLogAdapter() {
            @Override
            public boolean isLoggable(int priority, String tag) {
                return debuglog;
            }
        });
    }
}
