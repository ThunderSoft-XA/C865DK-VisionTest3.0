package com.thundercomm.eBox.Utils;

import android.graphics.Matrix;


public class ImageUtils {


    public static Matrix getTransformationMatrix(
            final int srcWidth,
            final int srcHeight,
            final int dstWidth,
            final int dstHeight) {
        final Matrix matrix = new Matrix();

        final int inWidth = srcWidth;
        final int inHeight = srcHeight;

        // Apply scaling if necessary.
        if (inWidth != dstWidth || inHeight != dstHeight) {
            final float scaleFactorX = dstWidth / (float) inWidth;
            final float scaleFactorY = dstHeight / (float) inHeight;

            // Scale exactly to fill dst from src.
            matrix.postScale(scaleFactorX, scaleFactorY);
        }

        return matrix;
    }
}

