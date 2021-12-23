package com.thundercomm.eBox;


public class Jni {
    static {
       // System.loadLibrary("nativedemo-lib");
    }

    public native int test();

    public static class Affinity {
        static {
            System.loadLibrary("affinitylib");
        }

        public static native int getCore();


        public static native void bindToCpu(int cpu);

        public static native void uubindFromCpu(int cpu);
    }
}

