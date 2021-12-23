package com.thundercomm.carDetect;

import com.thundercomm.eBox.carDetect.aipOcrClient;
import com.thundercomm.eBox.carDetect.carPlateLicenseDetect;

import org.junit.Before;
import org.junit.Test;

public class carPlateLicenseDetectTest {

    @Before
    public void setUp() throws Exception {
    }

    @Test
    public void detect() {
        aipOcrClient ocrClient = aipOcrClient.getInstance() ;
        carPlateLicenseDetect carDetect = new carPlateLicenseDetect(ocrClient.client);
        String path = "H:\\edgebox\\doc\\testpic\\test.jpg";
        carDetect.detect(path);

        System.out.println("init ok");
        return;
    }

    @Test
    public void detect1() {
    }
}