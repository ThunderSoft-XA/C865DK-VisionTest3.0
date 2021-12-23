package com.thundercomm.eBox.Adapter;

import android.annotation.SuppressLint;
import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.thundercomm.eBox.Constants.Constants;
import com.thundercomm.eBox.R;
import com.thundercomm.gateway.data.DeviceCollection;
import com.thundercomm.gateway.data.DeviceData;

import java.util.List;

/**
 * 设备信息适配器
 * @Describe
 */
public class RtspItemAdapter extends ArrayAdapter {
    private final int resourceId;

    public RtspItemAdapter(@NonNull Context context, int resource, @NonNull List<DeviceData> objects) {
        super(context, resource, objects);
        resourceId = resource;
    }

    @SuppressLint("SetTextI18n")
    @NonNull
    @Override
    public View getView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {
        DeviceData deviceData = (DeviceData) getItem(position);
        DeviceCollection deviceCollection = DeviceCollection.getInstance();
        DeviceCollection.getInstance().getAttributesValue(deviceData, "URL");
        @SuppressLint("ViewHolder") View view = LayoutInflater.from(getContext()).inflate(resourceId, null);
        TextView urlTv = view.findViewById(R.id.rtsp_item_Url);
        if (deviceData != null) {
            urlTv.setText("ID:" + (position + 1) + "  Video url: " + deviceCollection.getAttributesValue(deviceData, "URL"));
        }
        return view;
    }
}
