package com.magicmirror.app.engine;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

/**
 * InSwapper 人脸交换器 — 使用 inswapper_128_fp16.onnx 模型
 *
 * 关键流程：
 * 1. 从模型 initializer 中提取 emap 矩阵（带数据验证）
 * 2. 对 ArcFace embedding 做线性变换 latent = normalize(dot(embedding, emap))
 * 3. 对齐目标人脸到 128x128，运行 InSwapper 推理
 * 4. 将换脸结果通过逆仿射变换贴回原图，使用颜色校正 + 平滑遮罩混合
 *    （对齐桌面版 seamlessClone 的效果）
 */
public class FaceSwapper {
    private static final String TAG = "FaceSwapper";
    private static final String MODEL_NAME = "inswapper_128_fp16.onnx";
    private static final int INPUT_SIZE = 128;
    private static final int EMBEDDING_DIM = 512;

    private OrtSession session;
    private final OrtEnvironment env;

    private float[][] emap = null;
    private boolean emapExtracted = false;

    public FaceSwapper(OrtEnvironment env) {
        this.env = env;
    }

    public void loadModel(Context context, boolean useGpu) throws Exception {
        byte[] modelBytes = ModelUtils.loadModel(context, MODEL_NAME);

        try {
            emap = extractEmapFromOnnx(modelBytes);
            if (emap != null) {
                emapExtracted = true;
                Log.i(TAG, "成功从模型中提取 emap 矩阵 [" + emap.length + "x" + emap[0].length + "]");
            } else {
                Log.i(TAG, "模型中未找到 emap initializer，假设模型内部已处理");
            }
        } catch (Exception e) {
            Log.w(TAG, "提取 emap 失败: " + e.getMessage() + "，将直接传 embedding");
        }

        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        ModelUtils.configureSessionOptions(opts, useGpu, TAG);
        session = env.createSession(modelBytes, opts);
        Log.i(TAG, "人脸交换模型加载成功, 输入: " + session.getInputNames());
    }

    // ==================== emap 提取（带数据验证） ====================

    private float[][] extractEmapFromOnnx(byte[] modelBytes) {
        try {
            return parseEmapFromProtobuf(modelBytes);
        } catch (Exception e) {
            Log.w(TAG, "Protobuf 解析 emap 失败: " + e.getMessage());
            return null;
        }
    }

    private float[][] parseEmapFromProtobuf(byte[] data) {
        byte[] emapBytes = "emap".getBytes();
        int expectedRawSize = EMBEDDING_DIM * EMBEDDING_DIM * 4;

        for (int i = 0; i < data.length - emapBytes.length; i++) {
            if (!bytesMatch(data, i, emapBytes)) continue;
            if (i < 2) continue;
            int lenByte = data[i - 1] & 0xFF;
            int tagByte = data[i - 2] & 0xFF;
            if (tagByte != 0x0A || lenByte != emapBytes.length) continue;

            Log.d(TAG, "找到候选 'emap' 在位置: " + i);
            int searchEnd = Math.min(i + expectedRawSize + 4096, data.length);

            // 优先 raw_data (field 13, wire type 2 = tag 0x6A)
            float[][] result = searchFieldData(data, i, searchEnd, 0x6A, expectedRawSize);
            if (result != null && validateEmap(result)) return result;

            // 回退 float_data packed (field 5, wire type 2 = tag 0x2A)
            result = searchFieldData(data, i, searchEnd, 0x2A, expectedRawSize);
            if (result != null && validateEmap(result)) return result;
        }

        Log.d(TAG, "未在模型中找到 'emap' initializer");
        return null;
    }

    private float[][] searchFieldData(byte[] data, int start, int end, int targetTag, int expectedSize) {
        for (int i = start; i < end - 5; i++) {
            if ((data[i] & 0xFF) != targetTag) continue;
            int[] varintResult = readVarint(data, i + 1);
            if (varintResult == null) continue;
            int length = varintResult[0];
            int dataStart = varintResult[1];
            if (length == expectedSize && dataStart + length <= data.length) {
                return parseRawDataToEmap(data, dataStart, length);
            }
        }
        return null;
    }

    private boolean validateEmap(float[][] emap) {
        if (emap == null || emap.length != EMBEDDING_DIM || emap[0].length != EMBEDDING_DIM) return false;
        int sampleCount = 0;
        float sumAbs = 0;
        for (int i = 0; i < EMBEDDING_DIM; i += 32) {
            for (int j = 0; j < EMBEDDING_DIM; j += 32) {
                float v = emap[i][j];
                if (Float.isNaN(v) || Float.isInfinite(v)) return false;
                sumAbs += Math.abs(v);
                sampleCount++;
            }
        }
        float avgAbs = sumAbs / sampleCount;
        if (avgAbs < 0.001f || avgAbs > 50f) {
            Log.w(TAG, "emap 平均绝对值异常: " + avgAbs);
            return false;
        }
        return true;
    }

    private boolean bytesMatch(byte[] data, int offset, byte[] pattern) {
        for (int j = 0; j < pattern.length; j++) {
            if (data[offset + j] != pattern[j]) return false;
        }
        return true;
    }

    private int[] readVarint(byte[] data, int offset) {
        try {
            long result = 0;
            int shift = 0;
            int pos = offset;
            while (pos < data.length && shift < 35) {
                byte b = data[pos];
                result |= (long) (b & 0x7F) << shift;
                pos++;
                if ((b & 0x80) == 0) return new int[]{(int) result, pos};
                shift += 7;
            }
        } catch (Exception e) { /* ignore */ }
        return null;
    }

    private float[][] parseRawDataToEmap(byte[] data, int offset, int length) {
        float[][] result = new float[EMBEDDING_DIM][EMBEDDING_DIM];
        ByteBuffer bb = ByteBuffer.wrap(data, offset, length).order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer fb = bb.asFloatBuffer();
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            for (int j = 0; j < EMBEDDING_DIM; j++) {
                result[i][j] = fb.get();
            }
        }
        return result;
    }

    private float[] applyEmapTransform(float[] embedding) {
        if (emap == null) return embedding;
        float[] latent = new float[EMBEDDING_DIM];
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            float sum = 0;
            for (int i = 0; i < EMBEDDING_DIM; i++) {
                sum += embedding[i] * emap[i][j];
            }
            latent[j] = sum;
        }
        return ModelUtils.l2Normalize(latent);
    }

    // ==================== 换脸推理 ====================

    public Bitmap swapFace(Bitmap sourceImage, FaceDetector.DetectedFace targetFace,
                           float[] srcEmbedding) throws Exception {
        if (session == null) throw new IllegalStateException("模型未加载");

        // 1. 对齐目标人脸到 128x128
        Bitmap alignedTarget = ModelUtils.alignFace(sourceImage, targetFace.landmarks, INPUT_SIZE);

        // 2. 预处理: BGR [0, 255]
        float[][][][] faceData = ModelUtils.bitmapToBgrFloat(alignedTarget, INPUT_SIZE);

        // 3. 准备源特征向量，应用 emap 变换
        float[] transformedEmbedding = (emapExtracted && emap != null)
                ? applyEmapTransform(srcEmbedding) : srcEmbedding;

        float[][] embeddingData = new float[1][transformedEmbedding.length];
        System.arraycopy(transformedEmbedding, 0, embeddingData[0], 0, transformedEmbedding.length);

        // 4. 运行推理
        OnnxTensor faceTensor = OnnxTensor.createTensor(env, faceData);
        OnnxTensor embTensor = OnnxTensor.createTensor(env, embeddingData);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        String[] inputNames = session.getInputNames().toArray(new String[0]);
        inputs.put(inputNames[0], faceTensor);
        if (inputNames.length > 1) {
            inputs.put(inputNames[1], embTensor);
        }

        OrtSession.Result result = session.run(inputs);
        float[][][][] output = (float[][][][]) result.get(0).getValue();
        faceTensor.close();
        embTensor.close();
        result.close();

        // 5. 转换为 Bitmap
        Bitmap swappedFace = ModelUtils.bgrFloatToBitmap(output, INPUT_SIZE, INPUT_SIZE);

        // 6. 贴回原图（颜色校正 + 平滑遮罩，对齐桌面版 seamlessClone 效果）
        Bitmap resultImage = pasteBack(sourceImage, swappedFace, alignedTarget, targetFace.landmarks);

        alignedTarget.recycle();
        swappedFace.recycle();

        return resultImage;
    }

    // ==================== paste-back（对齐桌面版 seamlessClone 效果） ====================

    /**
     * 将换脸结果贴回原图。
     * 策略：
     * 1. 颜色校正：在对齐空间匹配原始人脸的颜色分布
     * 2. 创建平滑遮罩（矩形渐变，与桌面版一致）
     * 3. 逆变换到原图空间
     * 4. alpha 混合
     */
    private Bitmap pasteBack(Bitmap original, Bitmap swappedFace, Bitmap alignedOriginal,
                             float[][] landmarks) {
        Matrix alignMatrix = ModelUtils.getAlignMatrix(landmarks, INPUT_SIZE);
        Matrix inverseMatrix = new Matrix();
        if (!alignMatrix.invert(inverseMatrix)) {
            Log.w(TAG, "无法计算逆矩阵，使用简单贴回");
            return pasteBackSimple(original, swappedFace, landmarks);
        }

        int imgW = original.getWidth();
        int imgH = original.getHeight();

        // 1. 颜色校正
        Bitmap correctedFace = colorTransfer(alignedOriginal, swappedFace, INPUT_SIZE);

        // 2. 创建矩形渐变遮罩（与桌面版一致）
        Bitmap mask = createRectGradientMask(INPUT_SIZE, INPUT_SIZE);

        // 3. 逆变换到原图空间
        Bitmap warpedFace = Bitmap.createBitmap(imgW, imgH, Bitmap.Config.ARGB_8888);
        Canvas faceCanvas = new Canvas(warpedFace);
        faceCanvas.drawBitmap(correctedFace, inverseMatrix,
                new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG));

        Bitmap warpedMask = Bitmap.createBitmap(imgW, imgH, Bitmap.Config.ARGB_8888);
        Canvas maskCanvas = new Canvas(warpedMask);
        maskCanvas.drawBitmap(mask, inverseMatrix,
                new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG));

        // 4. alpha 混合
        Bitmap resultBmp = blendWithMask(original, warpedFace, warpedMask);

        correctedFace.recycle();
        mask.recycle();
        warpedFace.recycle();
        warpedMask.recycle();

        return resultBmp;
    }

    /**
     * 创建矩形渐变遮罩（与桌面版 _swap_single_face 的白色矩形遮罩 + seamlessClone 对齐）。
     * 中心区域为白色（完全使用换脸结果），边缘渐变到透明。
     * 渐变宽度约为尺寸的 12%，模拟 seamlessClone 的边缘过渡。
     */
    private Bitmap createRectGradientMask(int width, int height) {
        Bitmap mask = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height];

        // 渐变边距（从边缘向内渐变的像素数）
        int borderX = Math.max(4, (int) (width * 0.12f));
        int borderY = Math.max(4, (int) (height * 0.12f));

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // 计算到四条边的最小距离
                float dx = Math.min(x, width - 1 - x);
                float dy = Math.min(y, height - 1 - y);

                // 各方向的渐变因子
                float fx = (dx < borderX) ? dx / (float) borderX : 1.0f;
                float fy = (dy < borderY) ? dy / (float) borderY : 1.0f;

                // 取最小值（角落区域更透明）
                float alpha = Math.min(fx, fy);
                // 平滑曲线（smoothstep）
                alpha = alpha * alpha * (3f - 2f * alpha);

                int a = clamp(Math.round(alpha * 255), 0, 255);
                pixels[y * width + x] = (a << 24) | 0x00FFFFFF;
            }
        }

        mask.setPixels(pixels, 0, width, 0, 0, width, height);
        return mask;
    }

    /**
     * 使用遮罩做 alpha 混合：result = original * (1 - maskAlpha) + face * maskAlpha
     */
    private Bitmap blendWithMask(Bitmap original, Bitmap warpedFace, Bitmap warpedMask) {
        int w = original.getWidth();
        int h = original.getHeight();

        int[] origPixels = new int[w * h];
        int[] facePixels = new int[w * h];
        int[] maskPixels = new int[w * h];
        int[] resultPixels = new int[w * h];

        original.getPixels(origPixels, 0, w, 0, 0, w, h);
        warpedFace.getPixels(facePixels, 0, w, 0, 0, w, h);
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
            int fR = (facePixels[i] >> 16) & 0xFF;
            int fG = (facePixels[i] >> 8) & 0xFF;
            int fB = facePixels[i] & 0xFF;

            int rR = clamp(Math.round(oR * (1f - alpha) + fR * alpha), 0, 255);
            int rG = clamp(Math.round(oG * (1f - alpha) + fG * alpha), 0, 255);
            int rB = clamp(Math.round(oB * (1f - alpha) + fB * alpha), 0, 255);

            resultPixels[i] = 0xFF000000 | (rR << 16) | (rG << 8) | rB;
        }

        Bitmap result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        result.setPixels(resultPixels, 0, w, 0, 0, w, h);
        return result;
    }

    // ==================== 颜色校正（模拟 seamlessClone 的颜色融合） ====================

    private Bitmap colorTransfer(Bitmap source, Bitmap target, int size) {
        int[] srcPixels = new int[size * size];
        int[] tgtPixels = new int[size * size];
        source.getPixels(srcPixels, 0, size, 0, 0, size, size);
        target.getPixels(tgtPixels, 0, size, 0, 0, size, size);

        int margin = size / 6;
        float[] srcMean = new float[3], srcStd = new float[3];
        float[] tgtMean = new float[3], tgtStd = new float[3];
        computeRegionStats(srcPixels, size, margin, srcMean, srcStd);
        computeRegionStats(tgtPixels, size, margin, tgtMean, tgtStd);

        Bitmap result = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
        int[] resultPixels = new int[size * size];

        float blendFactor = 0.5f;
        for (int i = 0; i < size * size; i++) {
            int pixel = tgtPixels[i];
            int a = (pixel >> 24) & 0xFF;
            float r = (pixel >> 16) & 0xFF;
            float g = (pixel >> 8) & 0xFF;
            float b = pixel & 0xFF;

            r = transferChannel(r, tgtMean[0], tgtStd[0], srcMean[0], srcStd[0], blendFactor);
            g = transferChannel(g, tgtMean[1], tgtStd[1], srcMean[1], srcStd[1], blendFactor);
            b = transferChannel(b, tgtMean[2], tgtStd[2], srcMean[2], srcStd[2], blendFactor);

            resultPixels[i] = (a << 24)
                    | (clamp(Math.round(r), 0, 255) << 16)
                    | (clamp(Math.round(g), 0, 255) << 8)
                    | clamp(Math.round(b), 0, 255);
        }

        result.setPixels(resultPixels, 0, size, 0, 0, size, size);
        return result;
    }

    private float transferChannel(float value, float tgtMean, float tgtStd,
                                  float srcMean, float srcStd, float blend) {
        if (tgtStd < 1f) tgtStd = 1f;
        if (srcStd < 1f) srcStd = 1f;
        float transferred = (value - tgtMean) * (srcStd / tgtStd) + srcMean;
        return value * (1f - blend) + transferred * blend;
    }

    private void computeRegionStats(int[] pixels, int size, int margin,
                                    float[] mean, float[] std) {
        float sumR = 0, sumG = 0, sumB = 0;
        int count = 0;
        for (int y = margin; y < size - margin; y++) {
            for (int x = margin; x < size - margin; x++) {
                int pixel = pixels[y * size + x];
                sumR += (pixel >> 16) & 0xFF;
                sumG += (pixel >> 8) & 0xFF;
                sumB += pixel & 0xFF;
                count++;
            }
        }
        if (count == 0) count = 1;
        mean[0] = sumR / count;
        mean[1] = sumG / count;
        mean[2] = sumB / count;

        float varR = 0, varG = 0, varB = 0;
        for (int y = margin; y < size - margin; y++) {
            for (int x = margin; x < size - margin; x++) {
                int pixel = pixels[y * size + x];
                float dr = ((pixel >> 16) & 0xFF) - mean[0];
                float dg = ((pixel >> 8) & 0xFF) - mean[1];
                float db = (pixel & 0xFF) - mean[2];
                varR += dr * dr;
                varG += dg * dg;
                varB += db * db;
            }
        }
        std[0] = (float) Math.sqrt(varR / count);
        std[1] = (float) Math.sqrt(varG / count);
        std[2] = (float) Math.sqrt(varB / count);
    }

    // ==================== 简单贴回（后备方案） ====================

    private Bitmap pasteBackSimple(Bitmap original, Bitmap swappedFace, float[][] landmarks) {
        Bitmap result = original.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(result);

        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
        float maxX = Float.MIN_VALUE, maxY = Float.MIN_VALUE;
        for (float[] pt : landmarks) {
            minX = Math.min(minX, pt[0]);
            minY = Math.min(minY, pt[1]);
            maxX = Math.max(maxX, pt[0]);
            maxY = Math.max(maxY, pt[1]);
        }

        float faceW = maxX - minX;
        float faceH = maxY - minY;
        float expand = Math.max(faceW, faceH) * 0.3f;

        RectF destRect = new RectF(
                minX - expand, minY - expand,
                maxX + expand, maxY + expand
        );

        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
        canvas.drawBitmap(swappedFace, null, destRect, paint);

        return result;
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