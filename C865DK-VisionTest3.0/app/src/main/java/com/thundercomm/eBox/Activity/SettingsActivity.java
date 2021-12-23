package com.thundercomm.eBox.Activity;

import android.app.ActionBar;
import android.content.Intent;
import android.os.Bundle;
import android.view.MenuItem;

import androidx.appcompat.app.AppCompatActivity;
import androidx.preference.Preference;
import androidx.preference.PreferenceFragmentCompat;

import com.thundercomm.eBox.R;
import com.thundercomm.eBox.Utils.PreferencesUtil;

public class SettingsActivity extends AppCompatActivity {

    private String TAG = "SettingsActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.settings_activity);
        getSupportFragmentManager()
                .beginTransaction()
                .replace(R.id.settings, new SettingsFragment())
                .commit();
        ActionBar actionBar = getActionBar();
        if (actionBar != null) {
            actionBar.setDisplayHomeAsUpEnabled(true);
            actionBar.setBackgroundDrawable(getDrawable(R.drawable.background));
            actionBar.setTitle("返回");
        }
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            default:
                return super.onOptionsItemSelected(item);
        }
    }

    public static class SettingsFragment extends PreferenceFragmentCompat {
        private Preference preference;

        @Override
        public void onCreatePreferences(Bundle savedInstanceState, String rootKey) {
            setPreferencesFromResource(R.xml.root_preferences, rootKey);
            settingUrlMethodInit();

            preference = findPreference("rtsp");
            preference.setOnPreferenceClickListener(preference -> {
                Intent intent = new Intent(getContext(), RtspListActivity.class);
                startActivity(intent);
                return false;
            });

            Preference getUrlMethodPre = findPreference("get_url_type");
            if (null != getUrlMethodPre) {
                getUrlMethodPre.setOnPreferenceChangeListener((preference, newValue) -> {
                    PreferencesUtil.getInstance(getContext()).setPreferences(preference.getKey(), (String) newValue);
                    if ("by_Manual".equals(newValue)) {
                        findPreference("URL_Manual").setVisible(true);
                    } else {
                        findPreference("URL_Manual").setVisible(false);
                    }
                    return true;
                });
            }

            Preference setUrlEntry = findPreference("URL_Manual");
            if (setUrlEntry != null) {
                setUrlEntry.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener() {
                    @Override
                    public boolean onPreferenceClick(Preference preference) {
                        startActivity(new Intent(getActivity(), RtspManagerActivity.class));
                        return false;
                    }
                });
            }

        }

        private void settingUrlMethodInit() {
            findPreference("URL_Manual").setVisible("by_Manual".equals(PreferencesUtil.getInstance(getContext()).getPreferenceString("get_url_type")));
        }
    }

}