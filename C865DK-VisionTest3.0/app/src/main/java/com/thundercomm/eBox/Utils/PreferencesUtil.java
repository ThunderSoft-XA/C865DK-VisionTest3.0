package com.thundercomm.eBox.Utils;

import android.content.Context;
import android.content.SharedPreferences;
import android.preference.PreferenceManager;

public class PreferencesUtil {
    private PreferencesUtil(Context context) {
        mContext = context;
        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(mContext);
    }

    private static Context mContext;

    private SharedPreferences sharedPreferences;

    private static PreferencesUtil instance;

    private static final Object lock = new Object();

    public static PreferencesUtil getInstance(Context context) {
        if (instance == null) {
            synchronized (lock) {
                if (context != null) {
                    instance = new PreferencesUtil(context);
                } else {
                    return instance;
                }
            }
        }
        return instance;
    }

    public void setPreferences(String key, String value) {
        if (sharedPreferences != null) {
            SharedPreferences.Editor editor = sharedPreferences.edit();
            editor.putString(key, value);
            editor.commit();
        }
    }

    public String getPreferenceString(String key) {
        String value = "";
        if (sharedPreferences != null) {
            value = sharedPreferences.getString(key, "null");
        }
        return value;
    }

    public int getPreferenceInt(String key) {
        int value = 0;
        if (sharedPreferences != null) {
            value = sharedPreferences.getInt(key, 0);
        }
        return value;
    }

    public boolean getPreferenceBoolean(String key) {
        boolean b = false;
        if (sharedPreferences != null) {
            b = sharedPreferences.getBoolean(key, false);
        }
        return b;
    }


}
