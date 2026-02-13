package com.magicmirror.app.engine;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.Log;

import ai.onnxruntime.OrtSession;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;

/**
 * 模型加载和图像处理工具类
 *
 * 关键约定：insightface 系列模型（SCRFD / ArcFace / InSwapper / GFPGAN）
 * 全部使用 BGR 通道顺序（与 Python OpenCV 一致）。
 */
public class ModelUtils {
    private static final String TAG = "ModelUtils";

    // ==================== ArcFace 标准 112x112 对齐模板 ====================
    // 来源: insightface/utils/face_align.py  arcface_dst
    public static final float[][] ARCFACE_TEMPLATE_112 = {
        {38.2946f, 51.6963f},
        {73.5318f, 51.5014f},
        {56.0252f, 71.7366f},
        {41.5493f, 92.3655f},
        {70.7299f, 92.2041f}
    };

    // ArcFace 模板缩放到 512x512（用于增强器对齐）
    public static final float[][] ARCFACE_TEMPLATE_512;
    static {
        float scale = 512.0f / 112.0f;
        ARCFACE_TEMPLATE_512 = new float[5][2];
        for (int i = 0; i < 5; i++) {
            ARCFACE_TEMPLATE_512[i][0] = ARCFACE_TEMPLATE_112[i][0] * scale;
            ARCFACE_TEMPLATE_512[i][1] = ARCFACE_TEMPLATE_112[i][1] * scale;
        }
    }

    // ==================== 模型加载 ====================

    /** 统一模型加载入口：优先外部存储，回退 assets */
    public static byte[] loadModel(Context context, String modelName) throws Exception {
        return loadModelFromAssets(context, modelName);
    }

    public static byte[] loadModelFromAssets(Context context, String modelName) throws Exception {
        // 优先从外部存储的 models 目录加载（方便替换大模型）
        File externalModel = new File(context.getExternalFilesDir("models"), modelName);
        if (externalModel.exists()) {
            Log.i(TAG, "从外部存储加载模型: " + externalModel.getAbsolutePath());
            return readFileBytes(externalModel);
        }
        Log.i(TAG, "从 assets 加载模型: " + modelName);
        InputStream is = context.getAssets().open("models/" + modelName);
        return readStreamBytes(is);
    }

    private static byte[] readFileBytes(File file) throws Exception {
        FileInputStream fis = new FileInputStream(file);
        byte[] bytes = readStreamBytes(fis);
        fis.close();
        return bytes;
    }

    private static byte[] readStreamBytes(InputStream is) throws Exception {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int len;
        while ((len = is.read(buffer)) != -1) {
            bos.write(buffer, 0, len);
        }
        is.close();
        return bos.toByteArray();
    }

    /**
     * 配置 ONNX Runtime SessionOptions：尝试 NNAPI → XNNPACK → CPU 回退。
     * XNNPACK 是 CPU 上的高性能推理后端，比默认 CPU EP 快 2-3 倍。
     */
    public static void configureSessionOptions(OrtSession.SessionOptions opts, boolean useGpu, String tag) {
        if (useGpu) {
            // 优先尝试 NNAPI（利用 GPU/DSP/NPU）
            try {
                opts.addNnapi();
                Log.i(tag, "NNAPI 加速已启用");
                return;
            } catch (Exception e) {
                Log.w(tag, "NNAPI 不可用: " + e.getMessage());
            }
        }
        // 回退到 XNNPACK（CPU 优化后端）
        try {
            // XNNPACK 通过 ORT extensions 或内置支持
            // ORT Android 1.17+ 内置 XNNPACK EP
            opts.addXnnpack(java.util.Collections.emptyMap());
            Log.i(tag, "XNNPACK CPU 加速已启用");
        } catch (Exception e) {
            Log.w(tag, "XNNPACK 不可用，使用默认 CPU: " + e.getMessage());
        }
    }

    // ==================== 人脸对齐（Umeyama 5 点相似变换） ====================

    /**
     * 根据 5 个关键点计算相似变换矩阵（对齐到标准人脸模板）。
     * 使用 Umeyama 算法，利用全部 5 个点做最小二乘拟合，
     * 比只用 3 个点的 setPolyToPoly 更稳定、更准确。
     *
     * @param landmarks 检测到的 5 个关键点 [5][2]
     * @param targetSize 目标图像尺寸（如 112 / 128 / 512）
     * @return Android Matrix（从原图到对齐图的变换）
     */
    public static Matrix getAlignMatrix(float[][] landmarks, int targetSize) {
        float scale = targetSize / 112.0f;
        float[][] dst = new float[5][2];
        for (int i = 0; i < 5; i++) {
            dst[i][0] = ARCFACE_TEMPLATE_112[i][0] * scale;
            dst[i][1] = ARCFACE_TEMPLATE_112[i][1] * scale;
        }
        return estimateSimilarityTransform(landmarks, dst);
    }

    /**
     * Umeyama 相似变换估计（旋转 + 均匀缩放 + 平移）。
     * 参考: Shinji Umeyama, "Least-Squares Estimation of Transformation Parameters
     *       Between Two Point Patterns", IEEE TPAMI 1991.
     *
     * 输入 src[n][2], dst[n][2]，输出 Android Matrix 表示的 2x3 仿射矩阵。
     */
    private static Matrix estimateSimilarityTransform(float[][] src, float[][] dst) {
        int n = src.length;

        // 1. 计算质心
        float srcMeanX = 0, srcMeanY = 0, dstMeanX = 0, dstMeanY = 0;
        for (int i = 0; i < n; i++) {
            srcMeanX += src[i][0];
            srcMeanY += src[i][1];
            dstMeanX += dst[i][0];
            dstMeanY += dst[i][1];
        }
        srcMeanX /= n; srcMeanY /= n;
        dstMeanX /= n; dstMeanY /= n;

        // 2. 去质心
        float[][] srcCentered = new float[n][2];
        float[][] dstCentered = new float[n][2];
        for (int i = 0; i < n; i++) {
            srcCentered[i][0] = src[i][0] - srcMeanX;
            srcCentered[i][1] = src[i][1] - srcMeanY;
            dstCentered[i][0] = dst[i][0] - dstMeanX;
            dstCentered[i][1] = dst[i][1] - dstMeanY;
        }

        // 3. 计算 src 方差
        float srcVar = 0;
        for (int i = 0; i < n; i++) {
            srcVar += srcCentered[i][0] * srcCentered[i][0]
                    + srcCentered[i][1] * srcCentered[i][1];
        }
        srcVar /= n;

        // 4. 计算协方差矩阵 H = dst^T * src  (2x2)
        //    H = [ a  b ]
        //        [ c  d ]
        float a = 0, b = 0, c = 0, d = 0;
        for (int i = 0; i < n; i++) {
            a += dstCentered[i][0] * srcCentered[i][0];
            b += dstCentered[i][0] * srcCentered[i][1];
            c += dstCentered[i][1] * srcCentered[i][0];
            d += dstCentered[i][1] * srcCentered[i][1];
        }
        a /= n; b /= n; c /= n; d /= n;

        // 5. 2x2 SVD:  H = U * S * V^T
        //    对于 2x2 矩阵可以解析求解
        float[] svdResult = svd2x2(a, b, c, d);
        float u00 = svdResult[0], u01 = svdResult[1], u10 = svdResult[2], u11 = svdResult[3];
        float s0 = svdResult[4], s1 = svdResult[5];
        float v00 = svdResult[6], v01 = svdResult[7], v10 = svdResult[8], v11 = svdResult[9];

        // 6. R = U * V^T，处理反射情况
        float detUV = (u00 * u11 - u01 * u10) * (v00 * v11 - v01 * v10);
        float sign = detUV < 0 ? -1.0f : 1.0f;

        // D = diag(1, sign)
        float r00 = u00 * v00 + u01 * v10 * sign;
        float r01 = u00 * v01 + u01 * v11 * sign;
        float r10 = u10 * v00 + u11 * v10 * sign;
        float r11 = u10 * v01 + u11 * v11 * sign;

        // 7. scale = trace(D * S) / srcVar
        float traceDS = s0 + s1 * sign;
        float sc = (srcVar > 1e-10f) ? traceDS / srcVar : 1.0f;

        // 8. translation = dstMean - scale * R * srcMean
        float tx = dstMeanX - sc * (r00 * srcMeanX + r01 * srcMeanY);
        float ty = dstMeanY - sc * (r10 * srcMeanX + r11 * srcMeanY);

        // 9. 构建 Android Matrix
        //    [ sc*r00  sc*r01  tx ]
        //    [ sc*r10  sc*r11  ty ]
        //    [   0       0     1  ]
        Matrix matrix = new Matrix();
        float[] values = {
            sc * r00, sc * r01, tx,
            sc * r10, sc * r11, ty,
            0, 0, 1
        };
        matrix.setValues(values);
        return matrix;
    }

    /**
     * 解析求解 2x2 矩阵的 SVD。
     * 返回 [u00,u01,u10,u11, s0,s1, v00,v01,v10,v11]
     */
    private static float[] svd2x2(float a, float b, float c, float d) {
        // 使用 Jacobi 旋转方法
        float e = (a + d) / 2.0f;
        float f = (a - d) / 2.0f;
        float g = (c + b) / 2.0f;
        float h = (c - b) / 2.0f;

        float q = (float) Math.sqrt(e * e + h * h);
        float r = (float) Math.sqrt(f * f + g * g);

        float s0 = q + r;
        float s1 = Math.abs(q - r);

        float a1 = (float) Math.atan2(g, f);
        float a2 = (float) Math.atan2(h, e);

        float theta = (a2 - a1) / 2.0f;
        float phi = (a2 + a1) / 2.0f;

        float cosTheta = (float) Math.cos(theta);
        float sinTheta = (float) Math.sin(theta);
        float cosPhi = (float) Math.cos(phi);
        float sinPhi = (float) Math.sin(phi);

        // U = [[cos(phi), -sin(phi)], [sin(phi), cos(phi)]]
        // V^T = [[cos(theta), sin(theta)], [-sin(theta), cos(theta)]]
        // V = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
        return new float[] {
            cosPhi, -sinPhi, sinPhi, cosPhi,     // U
            s0, s1,                                // S
            cosTheta, -sinTheta, sinTheta, cosTheta // V
        };
    }

    /**
     * 使用仿射变换对齐人脸
     */
    public static Bitmap alignFace(Bitmap source, float[][] landmarks, int targetSize) {
        Matrix matrix = getAlignMatrix(landmarks, targetSize);
        Bitmap aligned = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(aligned);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
        canvas.drawBitmap(source, matrix, paint);
        return aligned;
    }

    // ==================== 原始 float[][] 仿射变换（用于增强器 paste-back）====================

    /**
     * Umeyama 相似变换，返回 float[2][3] 仿射矩阵（非 Android Matrix）。
     * 用于增强器的 warpAffine / invertAffine 流程。
     *
     * @param src 源关键点 [n][2]
     * @param dst 目标模板 [n][2]
     * @param estimateScale 是否估计缩放
     * @return float[2][3] 仿射矩阵
     */
    public static float[][] umeyamaTransform(float[][] src, float[][] dst, boolean estimateScale) {
        int n = src.length;
        float srcMx = 0, srcMy = 0, dstMx = 0, dstMy = 0;
        for (int i = 0; i < n; i++) {
            srcMx += src[i][0]; srcMy += src[i][1];
            dstMx += dst[i][0]; dstMy += dst[i][1];
        }
        srcMx /= n; srcMy /= n; dstMx /= n; dstMy /= n;

        float srcVar = 0;
        float a = 0, b = 0, c = 0, d = 0;
        for (int i = 0; i < n; i++) {
            float sx = src[i][0] - srcMx, sy = src[i][1] - srcMy;
            float dx = dst[i][0] - dstMx, dy = dst[i][1] - dstMy;
            srcVar += sx * sx + sy * sy;
            a += dx * sx; b += dx * sy; c += dy * sx; d += dy * sy;
        }
        srcVar /= n; a /= n; b /= n; c /= n; d /= n;

        float[] svd = svd2x2Internal(a, b, c, d);
        float u00 = svd[0], u01 = svd[1], u10 = svd[2], u11 = svd[3];
        float s0 = svd[4], s1 = svd[5];
        float v00 = svd[6], v01 = svd[7], v10 = svd[8], v11 = svd[9];

        float detUV = (u00*u11 - u01*u10) * (v00*v11 - v01*v10);
        float sign = detUV < 0 ? -1f : 1f;

        float r00 = u00*v00 + u01*v10*sign;
        float r01 = u00*v01 + u01*v11*sign;
        float r10 = u10*v00 + u11*v10*sign;
        float r11 = u10*v01 + u11*v11*sign;

        float sc = 1f;
        if (estimateScale && srcVar > 1e-10f) {
            sc = (s0 + s1 * sign) / srcVar;
        }

        float tx = dstMx - sc * (r00 * srcMx + r01 * srcMy);
        float ty = dstMy - sc * (r10 * srcMx + r11 * srcMy);

        return new float[][] {
            { sc * r00, sc * r01, tx },
            { sc * r10, sc * r11, ty }
        };
    }

    /**
     * 求 2x3 仿射矩阵的逆矩阵（仍为 2x3）。
     */
    public static float[][] invertAffine(float[][] m) {
        float a = m[0][0], b = m[0][1], tx = m[0][2];
        float c = m[1][0], d = m[1][1], ty = m[1][2];
        float det = a * d - b * c;
        if (Math.abs(det) < 1e-10f) throw new ArithmeticException("仿射矩阵不可逆");
        float invDet = 1f / det;
        float ia = d * invDet, ib = -b * invDet;
        float ic = -c * invDet, id = a * invDet;
        return new float[][] {
            { ia, ib, -(ia * tx + ib * ty) },
            { ic, id, -(ic * tx + id * ty) }
        };
    }

    /**
     * 使用 float[2][3] 仿射矩阵对 Bitmap 做仿射变换。
     * 输出指定尺寸的 Bitmap，超出范围的像素为透明。
     */
    public static Bitmap warpAffine(Bitmap src, float[][] mat, int outW, int outH) {
        // 将 float[2][3] 转为 Android Matrix
        Matrix m = new Matrix();
        float[] vals = {
            mat[0][0], mat[0][1], mat[0][2],
            mat[1][0], mat[1][1], mat[1][2],
            0, 0, 1
        };
        m.setValues(vals);

        Bitmap out = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(out);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
        canvas.drawBitmap(src, m, paint);
        return out;
    }

    /**
     * 计算两个矩形的 IoU（与 iou() 相同，提供别名以保持 API 一致性）
     */
    public static float calculateIoU(RectF a, RectF b) {
        return iou(a, b);
    }

    // 内部 SVD 方法（供 umeyamaTransform 使用）
    private static float[] svd2x2Internal(float a, float b, float c, float d) {
        float e = (a + d) / 2f, f = (a - d) / 2f;
        float g = (c + b) / 2f, h = (c - b) / 2f;
        float q = (float) Math.sqrt(e*e + h*h);
        float r = (float) Math.sqrt(f*f + g*g);
        float s0 = q + r, s1 = Math.abs(q - r);
        float a1 = (float) Math.atan2(g, f);
        float a2 = (float) Math.atan2(h, e);
        float theta = (a2 - a1) / 2f, phi = (a2 + a1) / 2f;
        float ct = (float) Math.cos(theta), st = (float) Math.sin(theta);
        float cp = (float) Math.cos(phi), sp = (float) Math.sin(phi);
        return new float[] { cp, -sp, sp, cp, s0, s1, ct, -st, st, ct };
    }

    // ==================== 裁剪 ====================

    public static Bitmap cropFace(Bitmap source, RectF box, float expandRatio) {
        int w = source.getWidth();
        int h = source.getHeight();
        float boxW = box.width();
        float boxH = box.height();
        float expandW = boxW * expandRatio;
        float expandH = boxH * expandRatio;

        int left = Math.max(0, (int) (box.left - expandW));
        int top = Math.max(0, (int) (box.top - expandH));
        int right = Math.min(w, (int) (box.right + expandW));
        int bottom = Math.min(h, (int) (box.bottom + expandH));

        return Bitmap.createBitmap(source, left, top, right - left, bottom - top);
    }

    // ==================== Bitmap ↔ float 数组转换（BGR 通道顺序） ====================

    /**
     * Bitmap → float[1][3][H][W]，BGR 顺序，像素值范围 [0, 255]（不归一化）。
     * 用于 InSwapper 等需要原始 BGR 像素值的模型。
     */
    public static float[][][][] bitmapToBgrFloat(Bitmap bitmap, int size) {
        Bitmap scaled = ensureSize(bitmap, size);
        float[][][][] data = new float[1][3][size][size];
        int[] pixels = new int[size * size];
        scaled.getPixels(pixels, 0, size, 0, 0, size, size);

        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                int pixel = pixels[y * size + x];
                int r = (pixel >> 16) & 0xFF;
                int g = (pixel >> 8) & 0xFF;
                int b = pixel & 0xFF;
                // BGR 顺序
                data[0][0][y][x] = b;
                data[0][1][y][x] = g;
                data[0][2][y][x] = r;
            }
        }
        if (scaled != bitmap) scaled.recycle();
        return data;
    }

    /**
     * Bitmap → float[1][3][H][W]，BGR 顺序，带均值/标准差归一化。
     * 公式: channel[i] = (pixel_bgr[i] - mean[i]) / std[i]
     *
     * @param mean BGR 顺序的均值
     * @param std  BGR 顺序的标准差
     */
    public static float[][][][] bitmapToBgrNormalized(Bitmap bitmap, int size,
                                                       float[] mean, float[] std) {
        Bitmap scaled = ensureSize(bitmap, size);
        float[][][][] data = new float[1][3][size][size];
        int[] pixels = new int[size * size];
        scaled.getPixels(pixels, 0, size, 0, 0, size, size);

        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                int pixel = pixels[y * size + x];
                int r = (pixel >> 16) & 0xFF;
                int g = (pixel >> 8) & 0xFF;
                int b = pixel & 0xFF;
                // BGR 顺序
                data[0][0][y][x] = (b - mean[0]) / std[0];
                data[0][1][y][x] = (g - mean[1]) / std[1];
                data[0][2][y][x] = (r - mean[2]) / std[2];
            }
        }
        if (scaled != bitmap) scaled.recycle();
        return data;
    }

    /**
     * float[1][3][H][W] (BGR) → Bitmap。
     * 像素值范围 [0, 255]。
     */
    public static Bitmap bgrFloatToBitmap(float[][][][] data, int width, int height) {
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int b = clamp((int) data[0][0][y][x], 0, 255);
                int g = clamp((int) data[0][1][y][x], 0, 255);
                int r = clamp((int) data[0][2][y][x], 0, 255);
                pixels[y * width + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        return bitmap;
    }

    /**
     * float[1][3][H][W] (BGR, [-1,1]) → Bitmap。
     * 反归一化: pixel = (value * 0.5 + 0.5) * 255
     */
    public static Bitmap bgrNormalizedToBitmap(float[][][][] data, int width, int height) {
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int b = clamp((int) ((data[0][0][y][x] * 0.5f + 0.5f) * 255), 0, 255);
                int g = clamp((int) ((data[0][1][y][x] * 0.5f + 0.5f) * 255), 0, 255);
                int r = clamp((int) ((data[0][2][y][x] * 0.5f + 0.5f) * 255), 0, 255);
                pixels[y * width + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        return bitmap;
    }

    // ==================== 保留旧接口（兼容） ====================

    /** @deprecated 使用 bitmapToBgrFloat 或 bitmapToBgrNormalized */
    public static float[][][][] bitmapToFloatArray(Bitmap bitmap, int size) {
        return bitmapToBgrFloat(bitmap, size);
    }

    /** @deprecated 使用 bitmapToBgrNormalized */
    public static float[][][][] bitmapToFloatArrayNormalized(Bitmap bitmap, int size,
                                                              float[] mean, float[] std) {
        return bitmapToBgrNormalized(bitmap, size, mean, std);
    }

    /** @deprecated 使用 bgrFloatToBitmap */
    public static Bitmap floatArrayToBitmap(float[][][][] data, int width, int height) {
        return bgrFloatToBitmap(data, width, height);
    }

    // ==================== 工具方法 ====================

    public static float[] l2Normalize(float[] vec) {
        float sum = 0;
        for (float v : vec) sum += v * v;
        float norm = (float) Math.sqrt(sum);
        if (norm < 1e-10f) return vec;
        float[] result = new float[vec.length];
        for (int i = 0; i < vec.length; i++) {
            result[i] = vec[i] / norm;
        }
        return result;
    }

    /**
     * 计算两个人脸框的 IoU
     */
    public static float iou(RectF a, RectF b) {
        float interLeft = Math.max(a.left, b.left);
        float interTop = Math.max(a.top, b.top);
        float interRight = Math.min(a.right, b.right);
        float interBottom = Math.min(a.bottom, b.bottom);
        float interArea = Math.max(0, interRight - interLeft) * Math.max(0, interBottom - interTop);
        float areaA = a.width() * a.height();
        float areaB = b.width() * b.height();
        float union = areaA + areaB - interArea;
        return union > 0 ? interArea / union : 0;
    }

    /**
     * 计算两个框中心点距离
     */
    public static float centerDistance(RectF a, RectF b) {
        float dx = a.centerX() - b.centerX();
        float dy = a.centerY() - b.centerY();
        return (float) Math.sqrt(dx * dx + dy * dy);
    }

    // ==================== Fix #11: 人脸框扩展和去重 ====================

    /**
     * 将人脸框扩展为正方形（与桌面版 _expand_square_box 一致）
     * @param box 原始人脸框
     * @param imgW 图像宽度
     * @param imgH 图像高度
     * @param scale 扩展比例（桌面版默认 1.35）
     * @param minSize 最小尺寸（桌面版默认 48）
     * @return 扩展后的正方形框，如果太小则返回 null
     */
    public static RectF expandSquareBox(RectF box, int imgW, int imgH, float scale, int minSize) {
        float w = box.width(), h = box.height();
        if (w < 1 || h < 1) return null;

        // 取较大边作为正方形边长
        float side = Math.max(w, h);
        // 扩展
        side *= scale;

        if (side < minSize) return null;

        // 以原框中心为中心
        float cx = box.centerX(), cy = box.centerY();
        float half = side / 2f;

        float left = cx - half;
        float top = cy - half;
        float right = cx + half;
        float bottom = cy + half;

        // 裁剪到图像边界
        left = Math.max(0, left);
        top = Math.max(0, top);
        right = Math.min(imgW, right);
        bottom = Math.min(imgH, bottom);

        // 再次检查尺寸
        if (right - left < minSize || bottom - top < minSize) return null;

        return new RectF(left, top, right, bottom);
    }

    /**
     * 去重人脸框（与桌面版 _dedupe_boxes 一致）
     * @param boxes 人脸框列表
     * @param iouThreshold IoU 阈值（桌面版默认 0.45）
     * @return 去重后的框列表
     */
    public static java.util.List<RectF> dedupeBoxes(java.util.List<RectF> boxes, float iouThreshold) {
        java.util.List<RectF> result = new java.util.ArrayList<>();
        boolean[] suppressed = new boolean[boxes.size()];
        for (int i = 0; i < boxes.size(); i++) {
            if (suppressed[i]) continue;
            result.add(boxes.get(i));
            for (int j = i + 1; j < boxes.size(); j++) {
                if (suppressed[j]) continue;
                if (iou(boxes.get(i), boxes.get(j)) > iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }
        return result;
    }

    // ==================== Fix #9: 图片格式兼容处理 ====================

    /**
     * 健壮的 Bitmap 加载（与桌面版 _read_image 一致）
     * 处理：16-bit 图像、灰度图、BGRA 格式
     * @param bitmap 原始 Bitmap
     * @return ARGB_8888 格式的 Bitmap
     */
    public static Bitmap ensureArgb8888(Bitmap bitmap) {
        if (bitmap == null) return null;

        // 如果已经是 ARGB_8888，直接返回
        if (bitmap.getConfig() == Bitmap.Config.ARGB_8888) return bitmap;

        // 转换为 ARGB_8888
        Bitmap converted = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        if (converted != bitmap) bitmap.recycle();
        return converted;
    }

    /**
     * 从字节数组健壮加载 Bitmap（处理各种格式）
     */
    public static Bitmap loadBitmapRobust(byte[] data) {
        if (data == null || data.length == 0) return null;

        android.graphics.BitmapFactory.Options opts = new android.graphics.BitmapFactory.Options();
        opts.inPreferredConfig = Bitmap.Config.ARGB_8888;
        opts.inMutable = true;

        Bitmap bmp = android.graphics.BitmapFactory.decodeByteArray(data, 0, data.length, opts);
        if (bmp == null) return null;

        return ensureArgb8888(bmp);
    }

    /**
     * 从文件路径健壮加载 Bitmap
     */
    public static Bitmap loadBitmapRobust(String path) {
        if (path == null || path.isEmpty()) return null;

        android.graphics.BitmapFactory.Options opts = new android.graphics.BitmapFactory.Options();
        opts.inPreferredConfig = Bitmap.Config.ARGB_8888;
        opts.inMutable = true;

        Bitmap bmp = android.graphics.BitmapFactory.decodeFile(path, opts);
        if (bmp == null) return null;

        return ensureArgb8888(bmp);
    }

    /**
     * 从 Context + Uri 健壮加载 Bitmap
     */
    public static Bitmap loadBitmapRobust(android.content.Context ctx, android.net.Uri uri) {
        if (ctx == null || uri == null) return null;
        try {
            java.io.InputStream is = ctx.getContentResolver().openInputStream(uri);
            if (is == null) return null;

            android.graphics.BitmapFactory.Options opts = new android.graphics.BitmapFactory.Options();
            opts.inPreferredConfig = Bitmap.Config.ARGB_8888;
            opts.inMutable = true;

            Bitmap bmp = android.graphics.BitmapFactory.decodeStream(is, null, opts);
            is.close();
            if (bmp == null) return null;

            return ensureArgb8888(bmp);
        } catch (Exception e) {
            Log.w(TAG, "加载图片失败: " + uri, e);
            return null;
        }
    }

    // ==================== Fix #10: 输出格式回退 ====================

    /**
     * 保存 Bitmap 到文件（与桌面版 _write_image 一致）
     * 尝试原格式，失败则回退到 PNG
     * @param bitmap 要保存的图片
     * @param outputPath 输出路径
     * @param originalPath 原始文件路径（用于推断格式），可为 null
     * @return 实际保存的文件路径
     */
    public static String saveBitmapRobust(Bitmap bitmap, String outputPath, String originalPath) {
        if (bitmap == null || outputPath == null) return null;

        // 推断原始格式
        Bitmap.CompressFormat format = Bitmap.CompressFormat.PNG;
        int quality = 95;
        if (originalPath != null) {
            String lower = originalPath.toLowerCase();
            if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
                format = Bitmap.CompressFormat.JPEG;
            } else if (lower.endsWith(".webp")) {
                format = Bitmap.CompressFormat.WEBP;
            }
            // PNG 是默认值
        }

        // 尝试用推断的格式保存
        try {
            java.io.FileOutputStream fos = new java.io.FileOutputStream(outputPath);
            boolean ok = bitmap.compress(format, quality, fos);
            fos.close();
            if (ok) return outputPath;
        } catch (Exception e) {
            Log.w(TAG, "用原格式保存失败: " + format, e);
        }

        // 回退到 PNG
        if (format != Bitmap.CompressFormat.PNG) {
            String pngPath = outputPath.replaceAll("\\.[^.]+$", ".png");
            try {
                java.io.FileOutputStream fos = new java.io.FileOutputStream(pngPath);
                boolean ok = bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
                fos.close();
                if (ok) return pngPath;
            } catch (Exception e) {
                Log.e(TAG, "PNG 回退也失败", e);
            }
        }

        return null;
    }

    // ==================== 工具方法 ====================

    private static Bitmap ensureSize(Bitmap bitmap, int size) {
        if (bitmap.getWidth() == size && bitmap.getHeight() == size) return bitmap;
        return Bitmap.createScaledBitmap(bitmap, size, size, true);
    }

    private static int clamp(int val, int min, int max) {
        return Math.max(min, Math.min(max, val));
    }
}