package com.magicmirror.app.engine;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * GFPGAN 人脸增强器 — 使用 gfpgan_1.4.onnx 模型
 *
 * 完整流程（与桌面版 _enhance_face 对齐）：
 *   1. 检测人脸关键点
 *   2. Umeyama 对齐到 512x512
 *   3. GFPGAN 推理增强
 *   4. 逆仿射变换增强结果到原图空间
 *   5. 在原图空间用矩形渐变 mask 混合（与桌面版一致）
 *
 * 关键改进（对齐桌面版）：
 * - 在原图空间做融合（先逆变换增强结果，再用遮罩混合），避免额外插值模糊
 * - 使用矩形渐变遮罩（从边缘向内渐变），与桌面版 seamlessClone 效果一致
 */
public class FaceEnhancer {
    private static final String TAG = "FaceEnhancer";
    private static final String MODEL_NAME = "gfpgan_1.4.onnx";
    private static final int INPUT_SIZE = 512;

    private OrtSession session;
    private final OrtEnvironment env;
    private boolean loaded = false;

    public FaceEnhancer(OrtEnvironment env) {
        this.env = env;
    }

    public void loadModel(Context context, boolean useGpu) throws Exception {
        byte[] modelBytes = ModelUtils.loadModel(context, MODEL_NAME);
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        ModelUtils.configureSessionOptions(opts, useGpu, TAG);
        session = env.createSession(modelBytes, opts);
        loaded = true;
        Log.i(TAG, "人脸增强模型加载成功");
    }

    public boolean isLoaded() {
        return loaded;
    }

    /**
     * 增强单张对齐后的 512x512 人脸图像（底层推理）
     */
    public Bitmap enhanceAligned(Bitmap faceBitmap) throws Exception {
        if (session == null) throw new IllegalStateException("模型未加载");

        // 预处理: BGR 通道，归一化到 [-1, 1]
        float[][][][] inputData = ModelUtils.bitmapToBgrNormalized(
                faceBitmap, INPUT_SIZE,
                new float[]{127.5f, 127.5f, 127.5f},
                new float[]{127.5f, 127.5f, 127.5f}
        );

        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(session.getInputNames().iterator().next(), inputTensor);

        OrtSession.Result result = session.run(inputs);

        // 输出: [1, 3, 512, 512]，BGR 通道，范围 [-1, 1]
        float[][][][] output = (float[][][][]) result.get(0).getValue();

        inputTensor.close();
        result.close();

        // 反归一化: [-1, 1] -> [0, 255]
        return ModelUtils.bgrNormalizedToBitmap(output, INPUT_SIZE, INPUT_SIZE);
    }

    /**
     * 增强原图中的单个人脸（完整流程：对齐→增强→逆变换→在原图空间混合）
     * 与桌面版 _enhance_face 流程对齐。
     *
     * @param image 原始完整图像
     * @param face 检测到的人脸（含关键点）
     * @return 增强后的完整图像
     */
    public Bitmap enhance(Bitmap image, FaceDetector.DetectedFace face) throws Exception {
        if (session == null) throw new IllegalStateException("模型未加载");
        if (face.landmarks == null || face.landmarks.length < 5) {
            Log.w(TAG, "人脸关键点不足，跳过增强");
            return image;
        }

        int imgW = image.getWidth();
        int imgH = image.getHeight();
        float[][] lm = face.landmarks;

        // 1. 计算仿射变换矩阵（对齐到 512x512 模板）
        float[][] affineMatrix = ModelUtils.umeyamaTransform(lm, ModelUtils.ARCFACE_TEMPLATE_512, true);

        // 2. 仿射变换对齐人脸到 512x512
        Bitmap aligned = ModelUtils.warpAffine(image, affineMatrix, INPUT_SIZE, INPUT_SIZE);

        // 3. GFPGAN 推理增强
        Bitmap enhanced = enhanceAligned(aligned);
        aligned.recycle();

        // 4. 计算逆仿射矩阵
        float[][] invMatrix;
        try {
            invMatrix = ModelUtils.invertAffine(affineMatrix);
        } catch (ArithmeticException e) {
            Log.w(TAG, "仿射矩阵不可逆，跳过增强");
            enhanced.recycle();
            return image;
        }

        // 5. 将增强后的人脸逆变换回原图尺寸
        Bitmap warpedBack = ModelUtils.warpAffine(enhanced, invMatrix, imgW, imgH);
        enhanced.recycle();

        // 6. 在原图空间做融合（与桌面版一致）
        //    创建矩形渐变遮罩（512x512 空间），逆变换到原图空间，然后 alpha 混合
        Bitmap mask512 = createRectGradientMask(INPUT_SIZE, INPUT_SIZE);
        Bitmap warpedMask = ModelUtils.warpAffine(mask512, invMatrix, imgW, imgH);
        mask512.recycle();

        Bitmap result = blendInOriginalSpace(image, warpedBack, warpedMask);
        warpedBack.recycle();
        warpedMask.recycle();

        return result;
    }

    /**
     * 增强原图中的所有人脸
     */
    public Bitmap enhanceAll(Bitmap image, List<FaceDetector.DetectedFace> faces) throws Exception {
        Bitmap result = image;
        for (FaceDetector.DetectedFace face : faces) {
            try {
                Bitmap enhanced = enhance(result, face);
                if (enhanced != result && result != image) {
                    result.recycle();
                }
                result = enhanced;
            } catch (Exception e) {
                Log.w(TAG, "增强单个人脸失败，跳过", e);
            }
        }
        return result;
    }

    /**
     * 创建矩形渐变遮罩（与桌面版 _enhance_face 的遮罩一致）。
     * 从边缘向内渐变，中心区域完全不透明。
     * 渐变宽度约为尺寸的 10%。
     */
    private Bitmap createRectGradientMask(int width, int height) {
        Bitmap mask = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height];

        int borderX = Math.max(4, (int) (width * 0.10f));
        int borderY = Math.max(4, (int) (height * 0.10f));

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float dx = Math.min(x, width - 1 - x);
                float dy = Math.min(y, height - 1 - y);

                float fx = (dx < borderX) ? dx / (float) borderX : 1.0f;
                float fy = (dy < borderY) ? dy / (float) borderY : 1.0f;

                float alpha = Math.min(fx, fy);
                // smoothstep 平滑曲线
                alpha = alpha * alpha * (3f - 2f * alpha);

                int a = clamp(Math.round(alpha * 255), 0, 255);
                pixels[y * width + x] = (a << 24) | 0x00FFFFFF;
            }
        }

        mask.setPixels(pixels, 0, width, 0, 0, width, height);
        return mask;
    }

    /**
     * 在原图空间做 alpha 混合（与桌面版一致）。
     * result = original * (1 - maskAlpha) + enhanced * maskAlpha
     */
    private Bitmap blendInOriginalSpace(Bitmap original, Bitmap warpedEnhanced, Bitmap warpedMask) {
        int w = original.getWidth();
        int h = original.getHeight();

        int[] origPixels = new int[w * h];
        int[] enhPixels = new int[w * h];
        int[] maskPixels = new int[w * h];
        int[] resultPixels = new int[w * h];

        original.getPixels(origPixels, 0, w, 0, 0, w, h);
        warpedEnhanced.getPixels(enhPixels, 0, w, 0, 0, w, h);
        warpedMask.getPixels(maskPixels, 0, w, 0, 0, w, h);

        for (int i = 0; i < w * h; i++) {
            int maskAlpha = (maskPixels[i] >> 24) & 0xFF;
            if (maskAlpha == 0) {
                resultPixels[i] = origPixels[i];
                continue;
            }

            float alpha = maskAlpha / 255f;
            int oR = (origPixels[i] >> 16) & 0xFF;
            int oG = (origPixels[i] >> 8) & 0xFF;
            int oB = origPixels[i] & 0xFF;
            int eR = (enhPixels[i] >> 16) & 0xFF;
            int eG = (enhPixels[i] >> 8) & 0xFF;
            int eB = enhPixels[i] & 0xFF;

            int rR = clamp(Math.round(oR * (1f - alpha) + eR * alpha), 0, 255);
            int rG = clamp(Math.round(oG * (1f - alpha) + eG * alpha), 0, 255);
            int rB = clamp(Math.round(oB * (1f - alpha) + eB * alpha), 0, 255);

            resultPixels[i] = 0xFF000000 | (rR << 16) | (rG << 8) | rB;
        }

        Bitmap result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        result.setPixels(resultPixels, 0, w, 0, 0, w, h);
        return result;
    }

    /**
     * 旧接口兼容 — 仅增强裁剪的人脸图（不推荐使用）
     * @deprecated 使用 enhance(Bitmap, DetectedFace) 代替
     */
    @Deprecated
    public Bitmap enhance(Bitmap faceBitmap) throws Exception {
        return enhanceAligned(faceBitmap);
    }

    private static int clamp(int val, int min, int max) {
        return Math.max(min, Math.min(max, val));
    }

    public void close() {
        try {
            if (session != null) session.close();
        } catch (Exception e) {
            Log.e(TAG, "关闭session失败", e);
        }
    }
}