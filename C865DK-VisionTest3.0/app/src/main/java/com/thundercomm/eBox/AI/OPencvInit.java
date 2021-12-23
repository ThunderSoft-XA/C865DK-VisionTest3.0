package com.thundercomm.eBox.AI;


public class OPencvInit {

    private static boolean isOPencvInit = false;

    public static boolean isLoaderOpenCV() {
        return isOPencvInit;
    }

    public static void setLoaderOpenCV(boolean suc) {
        isOPencvInit = suc;
    }
}
