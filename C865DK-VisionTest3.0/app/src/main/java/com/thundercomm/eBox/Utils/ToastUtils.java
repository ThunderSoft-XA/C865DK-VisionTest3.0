package com.thundercomm.eBox.Utils;

import android.content.Context;
import android.widget.Toast;



public class ToastUtils {

    public static void show(Context context, CharSequence text) {
        Toast.makeText(context, text, Toast.LENGTH_SHORT).show();
    }

}