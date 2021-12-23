package com.thundercomm.eBox.Activity;

import android.os.Bundle;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.widget.ListView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.gson.Gson;
import com.thundercomm.eBox.Constants.Constants;
import com.thundercomm.eBox.EventBusManager;
import com.thundercomm.eBox.Log.LogManager;
import com.thundercomm.eBox.Model.RtspItemCollection;
import com.thundercomm.eBox.R;
import com.thundercomm.eBox.Adapter.RtspItemAdapter;
import com.thundercomm.eBox.Utils.LogUtil;
import com.thundercomm.eBox.Utils.PreferencesUtil;

import org.greenrobot.eventbus.Subscribe;
import org.json.JSONException;
import org.json.JSONObject;

import static com.thundercomm.eBox.Constants.Constants.MAX_RTSP_NUM;

public class RtspListActivity extends AppCompatActivity {
    private ListView mListView;
    private String TAG = "RtspListActivity";
    private RtspItemAdapter mAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_rtsp_list);
        mAdapter = new RtspItemAdapter(RtspListActivity.this, R.layout.rtsp_item, RtspItemCollection.getInstance().getDeviceList());
        mListView = findViewById(R.id.lv_rtsp);
        mListView.setAdapter(mAdapter);
        EventBusManager.register(this);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.settingtoolbar_menu, menu);
        return super.onCreateOptionsMenu(menu);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.refuselist:
                Toast.makeText(this, "刷新", Toast.LENGTH_SHORT).show();
                break;
            case R.id.removelist:
                Toast.makeText(this, "clean list", Toast.LENGTH_SHORT).show();
        }
        return super.onOptionsItemSelected(item);
    }

}
