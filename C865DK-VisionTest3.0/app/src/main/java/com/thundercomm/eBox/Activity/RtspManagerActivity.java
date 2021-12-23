package com.thundercomm.eBox.Activity;

import android.app.AlertDialog;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.AdapterView;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import com.thundercomm.eBox.Adapter.RtspItemAdapter;
import com.thundercomm.eBox.Constants.Constants;
import com.thundercomm.eBox.Model.RtspItemCollection;
import com.thundercomm.eBox.R;
import com.thundercomm.eBox.Utils.LogUtil;
import com.thundercomm.gateway.data.DeviceCollection;
import com.thundercomm.gateway.data.DeviceData;
import com.thundercomm.gateway.utils.JsonTools;

public class RtspManagerActivity extends AppCompatActivity {
    private ListView mListView;
    private String TAG = "RtspManagerActivity";
    private RtspItemAdapter mAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_rtsp_list);
        mAdapter = new RtspItemAdapter(RtspManagerActivity.this, R.layout.rtsp_item, RtspItemCollection.getInstance().getDeviceList());
        mListView = findViewById(R.id.lv_rtsp);
        mListView.setAdapter(mAdapter);

        mListView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                showEditMenu(position);
            }
        });

        mListView.setOnItemLongClickListener(new AdapterView.OnItemLongClickListener() {
            @Override
            public boolean onItemLongClick(AdapterView<?> parent, View view, int position, long id) {
                LogUtil.i(TAG, "onItemLongClick ---> position :" + position);
                removeUrlItem(position);
                return true;
            }
        });
    }

    private void saveAndUpdata() {
        DeviceCollection.getInstance().writeDeviceCollection();
        mAdapter.notifyDataSetChanged();
    }

    private void showEditMenu(int position) {
        DeviceData deviceData = DeviceCollection.getInstance().getDeviceList().get(position);
        createEditUrlDialog(position, deviceData);
        //TODO：弹出编辑菜单
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.rtspmanagertoolbar_menu, menu);
        return super.onCreateOptionsMenu(menu);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.rtm_url_add:
                addNewUrlItem();
                break;
            case R.id.rtm_url_remove:
        }
        return super.onOptionsItemSelected(item);
    }

    /**
     * @Describe 移除一个Url地址流
     */
    private void removeUrlItem(int itemId) {
        DeviceCollection.getInstance().getDeviceList().remove(itemId);
        saveAndUpdata();
        LogUtil.i(TAG, "removeUrlItem ---> position :" + itemId);
    }

    /**
     * @Describe 添加一个Url地址流
     */
    private void addNewUrlItem() {
        LogUtil.i(TAG, "addNewUrlItem");
        if (mAdapter.getCount() >= Constants.MAX_RTSP_NUM) {
            Toast.makeText(this, "当前视频流数目已达上限...最大" + Constants.MAX_RTSP_NUM + "条", Toast.LENGTH_SHORT).show();
        } else {
            createNewUrlDialog();
        }
    }


    /**
     * @Describe 创建并显示一个添加视频流地址的对话框
     */
    private void createNewUrlDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        View view = LayoutInflater.from(this).inflate(R.layout.rtsp_edit_layout, null);
        final EditText editUrl = view.findViewById(R.id.ll_rtsp_etext);

        builder.setView(view);
        builder.setPositiveButton("OK", (dialog, which) -> {
            LogUtil.i(TAG, "OK..." + editUrl.getText().toString());
            if ("".equals(editUrl.getText().toString())) {
                Toast.makeText(getApplicationContext(), "url can not be empty!!", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(getApplicationContext(), "add: " + editUrl.getText(), Toast.LENGTH_SHORT).show();
                addNewRtspToCollection(editUrl.getText().toString(), false, false, true, false, false, false, false);
            }
        });
        builder.create().show();
    }

    private void createEditUrlDialog(int pos, DeviceData deviceData) {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        View view = LayoutInflater.from(this).inflate(R.layout.rtsp_edit_layout, null);
        final EditText editUrl = view.findViewById(R.id.ll_rtsp_etext);
        editUrl.setText(DeviceCollection.getInstance().getAttributesValue(deviceData, "URL"));
        builder.setView(view);
        builder.setPositiveButton("modify", (dialog, which) -> {
            LogUtil.i(TAG, "ok..." + editUrl.getText().toString());
            if ("".equals(editUrl.getText().toString())) {
                Toast.makeText(getApplicationContext(), "url can not be empty!!", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(getApplicationContext(), "modify: " + editUrl.getText(), Toast.LENGTH_SHORT).show();
                uploadRtspToCollection(pos, editUrl.getText().toString(), false, false, true, false, false, false, false);
            }
        });
        builder.setNeutralButton("delete", (dialog, which) -> {
            removeUrlItem(pos);
        });
        builder.create().show();
    }

    /**
     * @Describe 通过手动设置添加一个新的视频流配置
     */
    private void addNewRtspToCollection(String url, boolean isEnableFace, boolean isEnableFlame, boolean isEnableAgeGender, boolean isEnableDms, boolean checked, boolean enableFaceDetect_rdoChecked, boolean enabletraffic_rdoChecked) {
        DeviceCollection.getInstance().getDeviceList().add(DeviceData.builder().attributes(JsonTools.getKvEntries(
                JsonTools.newNode().
                        put("URL", url).
                        put("location", " ").
                        put(Constants.ENABLE_GESTURECLASSIFICATION_STR, String.valueOf(isEnableAgeGender)))).
                name("Camera_" + (DeviceCollection.getInstance().getDeviceList().size() + 1)).build());
        saveAndUpdata();
    }

    /**
     * @Describe 通过手动配置更新一个视频流配置
     */
    private void uploadRtspToCollection(int pos, String url, boolean isEnableFace, boolean isEnableFlame, boolean isEnableMultiObjectDetection, boolean isEnableDms, boolean checked, boolean enableFaceDetect_rdoChecked, boolean enabletraffic_rdoChecked) {
        String deviceName = DeviceCollection.getInstance().getDeviceList().get(pos).getName();
        DeviceCollection.getInstance().getDeviceList().remove(pos);
        DeviceCollection.getInstance().getDeviceList().add(pos, DeviceData.builder().attributes(JsonTools.getKvEntries(
                JsonTools.newNode().
                        put("URL", url).
                        put("location", " ").
                        put(Constants.ENABLE_GESTURECLASSIFICATION_STR, String.valueOf(isEnableMultiObjectDetection)))).
                name(deviceName).build());
        saveAndUpdata();
    }

    private int isVisible(boolean flag) {
        if (flag) {
            return View.VISIBLE;
        } else {
            return View.GONE;
        }
    }

}
