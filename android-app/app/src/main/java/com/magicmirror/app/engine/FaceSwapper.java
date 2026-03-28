package com.magicmirror.app.engine;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.Log;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
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
 * （对齐桌面版 seamlessClone 的效果）
 */
public class FaceSwapper {
    private static final String TAG = "FaceSwapper";
    private static final String MODEL_NAME = "inswapper_128_fp16.onnx";
    private static final int INPUT_SIZE = 128;
    private static final int EMBEDDING_DIM = 512;

    private static final int EMB_STRATEGY_EMAP = 0;
    private static final int EMB_STRATEGY_EMAP_TRANSPOSE = 1;
    private static final int EMB_STRATEGY_RAW = 2;

    private OrtSession session;
    private final OrtEnvironment env;

    private float[][] emap = null;
    private boolean emapExtracted = false;
    private String faceInputName = null;
    private String embeddingInputName = null;
    private String outputImageName = null;
    private int outputImageIndex = 0;

    private boolean embeddingStrategyResolved = false;
    private int embeddingStrategy = EMB_STRATEGY_EMAP;

    public FaceSwapper(OrtEnvironment env) {
        this.env = env;
    }

    public void loadModel(Context context, boolean useGpu) throws Exception {
        File modelFile = ModelUtils.prepareModelFile(context, MODEL_NAME);

        // emap 提取：优先走低内存文件扫描，避免大模型触发 OOM
        try {
            emap = extractEmapFromOnnxFile(modelFile);
            if (emap != null) {
                emapExtracted = true;
                Log.i(TAG, "成功从模型中提取 emap 矩阵 [" + emap.length + "x" + emap[0].length + "]");
            } else {
                Log.i(TAG, "模型中未找到 emap initializer，假设模型内部已处理");
            }
        } catch (Exception e) {
            Log.w(TAG, "跳过 emap 提取（不影响基本换脸）: " + e.getMessage());
        }

        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        ModelUtils.configureSessionOptions(opts, useGpu, TAG);
        session = env.createSession(modelFile.getAbsolutePath(), opts);
        resolveInputNames();
        resolveOutputName();
        Log.i(TAG, "人脸交换模型加载成功, 输入: " + session.getInputNames()
                + ", 输出: " + session.getOutputNames()
                + ", faceInput=" + faceInputName
                + ", embeddingInput=" + embeddingInputName
                + ", outputImage=" + outputImageName
                + ", outputIndex=" + outputImageIndex);
    }

    // ==================== emap 提取（带数据验证，低内存） ====================

    private float[][] extractEmapFromOnnxFile(File modelFile) {
        final int expectedRawSize = EMBEDDING_DIM * EMBEDDING_DIM * 4;
        final byte[] emapBytes = "emap".getBytes(StandardCharsets.US_ASCII);

        // 小模型直接内存解析
        try {
            if (modelFile.length() <= 64L * 1024L * 1024L) {
                byte[] modelBytes = ModelUtils.readSmallFileBytes(modelFile, 64L * 1024L * 1024L);
                return parseEmapFromProtobuf(modelBytes);
            }
        } catch (Exception e) {
            Log.w(TAG, "小模型内存解析 emap 失败，回退文件扫描: " + e.getMessage());
        }

        // 大模型：按块扫描，避免整体读入内存
        try (RandomAccessFile raf = new RandomAccessFile(modelFile, "r")) {
            long fileLen = raf.length();
            final int chunkSize = 4 * 1024 * 1024;
            final int overlap = Math.max(32, emapBytes.length + 16);

            byte[] chunk = new byte[chunkSize];
            byte[] tail = new byte[overlap];
            int tailLen = 0;
            long offset = 0L;

            while (offset < fileLen) {
                int toRead = (int) Math.min(chunkSize, fileLen - offset);
                raf.seek(offset);
                int read = raf.read(chunk, 0, toRead);
                if (read <= 0) {
                    break;
                }

                byte[] window = new byte[tailLen + read];
                if (tailLen > 0) {
                    System.arraycopy(tail, 0, window, 0, tailLen);
                }
                System.arraycopy(chunk, 0, window, tailLen, read);

                for (int i = 0; i <= window.length - emapBytes.length; i++) {
                    if (!bytesMatch(window, i, emapBytes)) {
                        continue;
                    }

                    long absPos = offset - tailLen + i;
                    if (absPos < 2) {
                        continue;
                    }

                    raf.seek(absPos - 2);
                    int tagByte = raf.read();
                    int lenByte = raf.read();
                    if (tagByte != 0x0A || lenByte != emapBytes.length) {
                        continue;
                    }

                    float[][] parsed = parseEmapNearPosition(raf, absPos, expectedRawSize, fileLen);
                    if (parsed != null && validateEmap(parsed)) {
                        Log.d(TAG, "通过文件扫描提取 emap 成功，位置: " + absPos);
                        return parsed;
                    }
                }

                tailLen = Math.min(overlap, window.length);
                System.arraycopy(window, window.length - tailLen, tail, 0, tailLen);
                offset += read;
            }
        } catch (Exception e) {
            Log.w(TAG, "文件扫描 emap 失败: " + e.getMessage());
        }

        return null;
    }

    private float[][] parseEmapNearPosition(RandomAccessFile raf, long startPos, int expectedRawSize, long fileLen)
            throws Exception {
        long maxWindow = expectedRawSize + 8192L;
        long available = Math.max(0L, fileLen - startPos);
        int readLen = (int) Math.min(maxWindow, available);
        if (readLen <= 0) {
            return null;
        }

        byte[] data = new byte[readLen];
        raf.seek(startPos);
        raf.readFully(data);

        // 优先 raw_data (TensorProto field 9, wire type 2 = tag 0x4A)
        float[][] result = searchFieldData(data, 0, data.length, 0x4A, expectedRawSize);
        if (result != null) {
            return result;
        }

        // 回退 float_data packed (TensorProto field 4, wire type 2 = tag 0x22)
        return searchFieldData(data, 0, data.length, 0x22, expectedRawSize);
    }

    private float[][] parseEmapFromProtobuf(byte[] data) {
        byte[] emapBytes = "emap".getBytes(StandardCharsets.US_ASCII);
        int expectedRawSize = EMBEDDING_DIM * EMBEDDING_DIM * 4;

        for (int i = 0; i < data.length - emapBytes.length; i++) {
            if (!bytesMatch(data, i, emapBytes))
                continue;
            if (i < 2)
                continue;
            int lenByte = data[i - 1] & 0xFF;
            int tagByte = data[i - 2] & 0xFF;
            if (tagByte != 0x0A || lenByte != emapBytes.length)
                continue;

            int searchEnd = Math.min(i + expectedRawSize + 4096, data.length);

            // 优先 raw_data (TensorProto field 9, wire type 2 = tag 0x4A)
            float[][] result = searchFieldData(data, i, searchEnd, 0x4A, expectedRawSize);
            if (result != null && validateEmap(result))
                return result;

            // 回退 float_data packed (TensorProto field 4, wire type 2 = tag 0x22)
            result = searchFieldData(data, i, searchEnd, 0x22, expectedRawSize);
            if (result != null && validateEmap(result))
                return result;
        }

        Log.d(TAG, "未在模型中找到 'emap' initializer");
        return null;
    }

    private float[][] searchFieldData(byte[] data, int start, int end, int targetTag, int expectedSize) {
        for (int i = start; i < end - 5; i++) {
            if ((data[i] & 0xFF) != targetTag)
                continue;
            int[] varintResult = readVarint(data, i + 1);
            if (varintResult == null)
                continue;
            int length = varintResult[0];
            int dataStart = varintResult[1];
            if (length == expectedSize && dataStart + length <= data.length) {
                return parseRawDataToEmap(data, dataStart, length);
            }
        }
        return null;
    }

    private boolean validateEmap(float[][] emap) {
        if (emap == null || emap.length != EMBEDDING_DIM || emap[0].length != EMBEDDING_DIM)
            return false;
        int sampleCount = 0;
        float sumAbs = 0;
        for (int i = 0; i < EMBEDDING_DIM; i += 32) {
            for (int j = 0; j < EMBEDDING_DIM; j += 32) {
                float v = emap[i][j];
                if (Float.isNaN(v) || Float.isInfinite(v))
                    return false;
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
            if (data[offset + j] != pattern[j])
                return false;
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
                if ((b & 0x80) == 0)
                    return new int[] { (int) result, pos };
                shift += 7;
            }
        } catch (Exception e) {
            /* ignore */ }
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
        if (emap == null)
            return embedding;
        float[] latent = new float[EMBEDDING_DIM];
        // latent = embedding(row) * emap
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            float sum = 0;
            for (int i = 0; i < EMBEDDING_DIM; i++) {
                sum += embedding[i] * emap[i][j];
            }
            latent[j] = sum;
        }
        return ModelUtils.l2Normalize(latent);
    }

    private float[] applyEmapTransformTransposed(float[] embedding) {
        if (emap == null)
            return embedding;
        float[] latent = new float[EMBEDDING_DIM];
        // 备用方向：latent = emap * embedding(col)
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            float sum = 0;
            for (int j = 0; j < EMBEDDING_DIM; j++) {
                sum += emap[i][j] * embedding[j];
            }
            latent[i] = sum;
        }
        return ModelUtils.l2Normalize(latent);
    }

    private float[][] toEmbeddingBatch(float[] embedding) {
        float[] normed = ModelUtils.l2Normalize(embedding);
        float[][] out = new float[1][normed.length];
        System.arraycopy(normed, 0, out[0], 0, normed.length);
        return out;
    }

    private String embeddingStrategyName(int strategy) {
        switch (strategy) {
            case EMB_STRATEGY_EMAP_TRANSPOSE:
                return "emap_transpose";
            case EMB_STRATEGY_RAW:
                return "raw";
            case EMB_STRATEGY_EMAP:
            default:
                return "emap";
        }
    }

    private float[] applyEmbeddingStrategy(float[] srcEmbedding, int strategy) {
        switch (strategy) {
            case EMB_STRATEGY_EMAP_TRANSPOSE:
                return applyEmapTransformTransposed(srcEmbedding);
            case EMB_STRATEGY_RAW:
                return ModelUtils.l2Normalize(srcEmbedding);
            case EMB_STRATEGY_EMAP:
            default:
                return applyEmapTransform(srcEmbedding);
        }
    }

    private float outputQuality(OutputStats s) {
        // 简单质量分：动态范围 + 标准差，惩罚近乎常量输出
        float q = s.range + s.std * 0.8f;
        if (s.range < 0.02f) {
            q -= 10f;
        }
        if (Math.abs(s.max - s.min) < 1e-6f) {
            q -= 10f;
        }
        return q;
    }

    private int resolveEmbeddingStrategy(Bitmap alignedTarget, float[] srcEmbedding) {
        // 对齐桌面版行为：优先固定使用 emap 线性变换；仅当 emap 不可用时回退 raw。
        // 避免运行时策略探测导致同一设备/素材间结果漂移。
        if (emapExtracted && emap != null) {
            return EMB_STRATEGY_EMAP;
        }
        return EMB_STRATEGY_RAW;
    }

    // ==================== 换脸推理 ====================

    public Bitmap swapFace(Bitmap sourceImage, FaceDetector.DetectedFace targetFace,
            float[] srcEmbedding) throws Exception {
        if (session == null)
            throw new IllegalStateException("模型未加载");

        if (targetFace == null || targetFace.landmarks == null || targetFace.landmarks.length < 5) {
            throw new IllegalArgumentException("目标人脸关键点无效，无法执行换脸");
        }

        // 1. 对齐目标人脸到 128x128
        Bitmap alignedTarget = ModelUtils.alignFace(sourceImage, targetFace.landmarks, INPUT_SIZE);

        // 2. 源特征向量策略（emap / emap转置 / raw）一次探测并缓存
        if (!embeddingStrategyResolved) {
            embeddingStrategy = resolveEmbeddingStrategy(alignedTarget, srcEmbedding);
            embeddingStrategyResolved = true;
            Log.w(TAG, "InSwapper embedding策略已锁定(桌面对齐): " + embeddingStrategyName(embeddingStrategy));
        }

        float[] chosenEmbedding = applyEmbeddingStrategy(srcEmbedding, embeddingStrategy);
        float[][] embeddingData = toEmbeddingBatch(chosenEmbedding);

        // 3. 运行推理（桌面对齐：固定 [0,1] 输入，不做动态输入尺度切换）
        InferenceResult infer = runInferenceDesktopLike(alignedTarget, embeddingData);
        float[][][][] output = infer.output;
        OutputStats stats = infer.stats;

        // 4. 转换为 Bitmap
        Bitmap swappedFace = decodeSwapOutput(output, stats);

        // 5. 颜色校正 + 贴回原图
        Bitmap harmonizedFace = colorTransfer(alignedTarget, swappedFace, INPUT_SIZE);
        Bitmap resultImage = pasteBack(sourceImage, harmonizedFace, targetFace.landmarks);

        alignedTarget.recycle();
        swappedFace.recycle();
        harmonizedFace.recycle();

        return resultImage;
    }

    // ==================== paste-back（对齐桌面版） ====================

    /**
     * 将换脸结果贴回原图。
     * 策略：
     * 1. 创建平滑遮罩（矩形渐变）
     * 2. 逆变换到原图空间
     * 3. alpha 混合
     */
    private Bitmap pasteBack(Bitmap original, Bitmap swappedFace, float[][] landmarks) {
        Matrix alignMatrix = ModelUtils.getAlignMatrix(landmarks, INPUT_SIZE);
        Matrix inverseMatrix = new Matrix();
        if (!alignMatrix.invert(inverseMatrix)) {
            Log.w(TAG, "无法计算逆矩阵，使用简单贴回");
            return pasteBackSimple(original, swappedFace, landmarks);
        }

        int imgW = original.getWidth();
        int imgH = original.getHeight();

        // 1. 创建矩形渐变遮罩（与桌面版一致）
        Bitmap mask = createRectGradientMask(INPUT_SIZE, INPUT_SIZE);

        // 2. 逆变换到原图空间
        Bitmap warpedFace = Bitmap.createBitmap(imgW, imgH, Bitmap.Config.ARGB_8888);
        Canvas faceCanvas = new Canvas(warpedFace);
        faceCanvas.drawBitmap(swappedFace, inverseMatrix,
                new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG));

        Bitmap warpedMask = Bitmap.createBitmap(imgW, imgH, Bitmap.Config.ARGB_8888);
        Canvas maskCanvas = new Canvas(warpedMask);
        maskCanvas.drawBitmap(mask, inverseMatrix,
                new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG));

        // 3. alpha 混合
        Bitmap resultBmp = blendWithMask(original, warpedFace, warpedMask);

        mask.recycle();
        warpedFace.recycle();
        warpedMask.recycle();

        return resultBmp;
    }

    /**
     * 创建矩形渐变遮罩（与桌面版一致）。
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
                alpha = alpha * alpha * (3f - 2f * alpha); // smoothstep

                int a = clamp(Math.round(alpha * 255f), 0, 255);
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
        if (tgtStd < 1f)
            tgtStd = 1f;
        if (srcStd < 1f)
            srcStd = 1f;
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
        if (count == 0)
            count = 1;
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

    // ==================== 推理输入输出适配 ====================

    private static class OutputStats {
        float min = Float.MAX_VALUE;
        float max = -Float.MAX_VALUE;
        float range = 0f;
        float mean = 0f;
        float std = 0f;
        int count = 0;
    }

    private static class InferenceResult {
        float[][][][] output;
        OutputStats stats;
    }

    private void resolveInputNames() {
        faceInputName = null;
        embeddingInputName = null;

        try {
            for (Map.Entry<String, NodeInfo> e : session.getInputInfo().entrySet()) {
                if (!(e.getValue().getInfo() instanceof TensorInfo)) {
                    continue;
                }
                TensorInfo ti = (TensorInfo) e.getValue().getInfo();
                long[] shape = ti.getShape();
                String name = e.getKey();
                String lower = name == null ? "" : name.toLowerCase();

                if (shape != null && shape.length >= 4 && faceInputName == null) {
                    faceInputName = name;
                    continue;
                }
                if (shape != null && shape.length == 2 && embeddingInputName == null) {
                    embeddingInputName = name;
                    continue;
                }

                if (faceInputName == null
                        && (lower.contains("img") || lower.contains("face") || lower.contains("target"))) {
                    faceInputName = name;
                } else if (embeddingInputName == null
                        && (lower.contains("emb") || lower.contains("id") || lower.contains("source")
                                || lower.contains("latent"))) {
                    embeddingInputName = name;
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "解析输入信息失败，回退默认输入顺序: " + e.getMessage());
        }

        if (faceInputName == null || (embeddingInputName == null && session.getInputNames().size() > 1)) {
            String first = null;
            String second = null;
            for (String name : session.getInputNames()) {
                if (first == null) {
                    first = name;
                } else if (second == null) {
                    second = name;
                    break;
                }
            }
            if (faceInputName == null) {
                faceInputName = first;
            }
            if (embeddingInputName == null && second != null && !second.equals(faceInputName)) {
                embeddingInputName = second;
            }
            if (embeddingInputName == null && first != null && !first.equals(faceInputName)
                    && session.getInputNames().size() > 1) {
                embeddingInputName = first;
            }
        }
    }

    private float[][][][] runSwapInference(Bitmap alignedTarget, float[][] embeddingData, boolean normalizedInput)
            throws Exception {
        // InSwapper 与 insightface 对齐：RGB 输入，默认归一化到 [0,1]
        float[][][][] faceData = normalizedInput
                ? ModelUtils.bitmapToRgbNormalized(alignedTarget, INPUT_SIZE,
                        new float[] { 0f, 0f, 0f }, new float[] { 255f, 255f, 255f })
                : ModelUtils.bitmapToRgbFloat(alignedTarget, INPUT_SIZE);

        try (OnnxTensor faceTensor = OnnxTensor.createTensor(env, faceData);
                OnnxTensor embTensor = OnnxTensor.createTensor(env, embeddingData)) {

            Map<String, OnnxTensor> inputs = new HashMap<>();
            if (faceInputName != null) {
                inputs.put(faceInputName, faceTensor);
            }
            if (embeddingInputName != null) {
                inputs.put(embeddingInputName, embTensor);
            }

            if (inputs.isEmpty()) {
                String[] names = session.getInputNames().toArray(new String[0]);
                if (names.length > 0) {
                    inputs.put(names[0], faceTensor);
                }
                if (names.length > 1) {
                    inputs.put(names[1], embTensor);
                }
            } else if (inputs.size() == 1 && session.getInputNames().size() > 1) {
                // 若只识别到一个输入名，补齐另一个，避免 set 顺序不确定造成错误映射
                for (String n : session.getInputNames()) {
                    if (inputs.containsKey(n)) {
                        continue;
                    }
                    inputs.put(n, inputs.containsValue(faceTensor) ? embTensor : faceTensor);
                    break;
                }
            }

            Log.d(TAG, "InSwapper.run: normalized=" + normalizedInput
                    + ", faceInput=" + faceInputName
                    + ", embInput=" + embeddingInputName
                    + ", output=" + outputImageName
                    + ", outputIndex=" + outputImageIndex);

            try (OrtSession.Result result = session.run(inputs)) {
                return selectSwapOutput(result);
            }
        }
    }

    private void resolveOutputName() {
        outputImageName = null;
        outputImageIndex = 0;

        try {
            String bestName = null;
            int bestScore = Integer.MIN_VALUE;
            for (Map.Entry<String, NodeInfo> e : session.getOutputInfo().entrySet()) {
                if (!(e.getValue().getInfo() instanceof TensorInfo)) {
                    continue;
                }
                TensorInfo ti = (TensorInfo) e.getValue().getInfo();
                long[] shape = ti.getShape();
                if (shape == null) {
                    continue;
                }

                int score = 0;
                if (shape.length == 4) {
                    score += 100;
                    long c = shape[1];
                    long h = shape[2];
                    long w = shape[3];
                    if (c == 3) {
                        score += 80;
                    }
                    if (h == INPUT_SIZE && w == INPUT_SIZE) {
                        score += 80;
                    }
                    if (h > 0 && w > 0) {
                        score += (int) Math.min(10_000L, h * w / 16L);
                    }
                } else if (shape.length == 3) {
                    score += 20;
                }

                String lower = e.getKey() == null ? "" : e.getKey().toLowerCase();
                if (lower.contains("out") || lower.contains("image") || lower.contains("swap")) {
                    score += 20;
                }

                if (score > bestScore) {
                    bestScore = score;
                    bestName = e.getKey();
                }
            }

            if (bestName == null && !session.getOutputNames().isEmpty()) {
                bestName = session.getOutputNames().iterator().next();
            }
            outputImageName = bestName;

            int idx = 0;
            for (String n : session.getOutputNames()) {
                if (n != null && n.equals(outputImageName)) {
                    outputImageIndex = idx;
                    break;
                }
                idx++;
            }
        } catch (Exception e) {
            Log.w(TAG, "解析输出信息失败，回退 outputIndex=0: " + e.getMessage());
            outputImageName = null;
            outputImageIndex = 0;
        }
    }

    private float[][][][] selectSwapOutput(OrtSession.Result result) throws Exception {
        if (result == null || result.size() <= 0) {
            throw new IllegalStateException("InSwapper 无输出");
        }

        if (outputImageIndex >= 0 && outputImageIndex < result.size()) {
            try {
                Object v = result.get(outputImageIndex).getValue();
                if (v instanceof float[][][][]) {
                    return (float[][][][]) v;
                }
            } catch (Exception ignored) {
            }
        }

        float[][][][] best = null;
        int bestScore = Integer.MIN_VALUE;

        for (int i = 0; i < result.size(); i++) {
            Object v;
            try {
                v = result.get(i).getValue();
            } catch (Exception e) {
                continue;
            }
            if (!(v instanceof float[][][][])) {
                continue;
            }
            float[][][][] arr = (float[][][][]) v;
            if (arr.length == 0 || arr[0] == null) {
                continue;
            }

            int c = arr[0].length;
            int h = (c > 0 && arr[0][0] != null) ? arr[0][0].length : 0;
            int w = (h > 0 && arr[0][0][0] != null) ? arr[0][0][0].length : 0;

            int score = 0;
            if (c == 3) {
                score += 80;
            }
            if (h == INPUT_SIZE && w == INPUT_SIZE) {
                score += 80;
            }
            if (h > 0 && w > 0) {
                score += Math.min(10_000, (h * w) / 16);
            }
            if (i == outputImageIndex) {
                score += 20;
            }

            if (score > bestScore) {
                bestScore = score;
                best = arr;
            }
        }

        if (best != null) {
            return best;
        }

        Object first = result.get(0).getValue();
        if (first instanceof float[][][][]) {
            return (float[][][][]) first;
        }

        throw new IllegalStateException("无法从 InSwapper 输出中解析图像张量，outputCount=" + result.size());
    }

    private InferenceResult runInferenceDesktopLike(Bitmap alignedTarget, float[][] embeddingData)
            throws Exception {
        InferenceResult out = new InferenceResult();

        float[][][][] output = runSwapInference(alignedTarget, embeddingData, true);
        OutputStats stats = analyzeOutput(output);

        out.output = output;
        out.stats = stats;
        return out;
    }

    private OutputStats analyzeOutput(float[][][][] output) {
        OutputStats s = new OutputStats();
        if (output == null || output.length == 0 || output[0] == null || output[0].length < 3) {
            s.min = 0f;
            s.max = 0f;
            s.range = 0f;
            return s;
        }

        double sum = 0.0;
        double sumSq = 0.0;

        for (int c = 0; c < Math.min(3, output[0].length); c++) {
            float[][] plane = output[0][c];
            if (plane == null) {
                continue;
            }
            for (int y = 0; y < plane.length; y++) {
                float[] row = plane[y];
                if (row == null) {
                    continue;
                }
                for (int x = 0; x < row.length; x++) {
                    float v = row[x];
                    if (Float.isNaN(v) || Float.isInfinite(v)) {
                        continue;
                    }
                    if (v < s.min)
                        s.min = v;
                    if (v > s.max)
                        s.max = v;
                    sum += v;
                    sumSq += (double) v * (double) v;
                    s.count++;
                }
            }
        }

        if (s.min == Float.MAX_VALUE || s.max == -Float.MAX_VALUE) {
            s.min = 0f;
            s.max = 0f;
        }
        s.range = s.max - s.min;

        if (s.count > 0) {
            s.mean = (float) (sum / s.count);
            double var = sumSq / s.count - (double) s.mean * (double) s.mean;
            s.std = (float) Math.sqrt(Math.max(0.0, var));
        } else {
            s.mean = 0f;
            s.std = 0f;
        }

        Log.d(TAG, "InSwapper 输出统计: min=" + s.min + ", max=" + s.max
                + ", range=" + s.range + ", mean=" + s.mean + ", std=" + s.std
                + ", count=" + s.count);
        return s;
    }

    private Bitmap decodeSwapOutput(float[][][][] output, OutputStats stats) {
        if (stats.max <= 1.25f && stats.min >= -1.25f) {
            // 常见情况：输出在 [-1,1] 或 [0,1]
            if (stats.min < -0.05f) {
                return ModelUtils.rgbNormalizedToBitmap(output, INPUT_SIZE, INPUT_SIZE);
            }
            return rgbUnitToBitmap(output, INPUT_SIZE, INPUT_SIZE);
        }

        if (stats.max <= 255.5f && stats.min >= -0.5f) {
            return ModelUtils.rgbFloatToBitmap(output, INPUT_SIZE, INPUT_SIZE);
        }

        // 极端异常输出：避免 min-max 拉伸导致整体“怪异偏色”，改为直接裁剪到合法范围
        return rgbClampedToBitmap(output, INPUT_SIZE, INPUT_SIZE);
    }

    private Bitmap rgbUnitToBitmap(float[][][][] data, int width, int height) {
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = clamp(Math.round(data[0][0][y][x] * 255f), 0, 255);
                int g = clamp(Math.round(data[0][1][y][x] * 255f), 0, 255);
                int b = clamp(Math.round(data[0][2][y][x] * 255f), 0, 255);
                pixels[y * width + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        return bitmap;
    }

    private Bitmap rgbClampedToBitmap(float[][][][] data, int width, int height) {
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float rv = data[0][0][y][x];
                float gv = data[0][1][y][x];
                float bv = data[0][2][y][x];

                int r = clamp(Math.round(rv), 0, 255);
                int g = clamp(Math.round(gv), 0, 255);
                int b = clamp(Math.round(bv), 0, 255);

                pixels[y * width + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        return bitmap;
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
                maxX + expand, maxY + expand);

        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
        canvas.drawBitmap(swappedFace, null, destRect, paint);

        return result;
    }

    private static int clamp(int val, int min, int max) {
        return Math.max(min, Math.min(max, val));
    }

    public void close() {
        try {
            if (session != null)
                session.close();
        } catch (Exception e) {
            Log.e(TAG, "关闭session失败", e);
        }
    }
}