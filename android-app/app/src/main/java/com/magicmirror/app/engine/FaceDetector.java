package com.magicmirror.app.engine;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * SCRFD 人脸检测器 — 使用 scrfd_2.5g.onnx 模型
 * 输入: BGR 图像 640x640，减均值除标准差
 * 输出: 多 stride (8/16/32) 的 scores + bboxes + landmarks
 */
public class FaceDetector {
    private static final String TAG = "FaceDetector";
    private static final String MODEL_NAME = "scrfd_2.5g.onnx";
    private static final int INPUT_SIZE = 640;
    private static final float CONF_THRESHOLD = 0.5f;
    private static final float NMS_THRESHOLD = 0.4f;
    private static final int MAX_OUTPUT_FACES = 20;
    private static final float MIN_FACE_SIDE_RATIO = 0.02f;
    private static final float MIN_FACE_SIDE_PX = 18f;
    private static final float MIN_FACE_AREA_RATIO = 0.00035f;
    private static final float MAX_FACE_AREA_RATIO = 0.60f;
    private static final float MIN_FACE_ASPECT = 0.55f;
    private static final float MAX_FACE_ASPECT = 1.85f;

    private OrtSession session;
    private final OrtEnvironment env;

    public FaceDetector(OrtEnvironment env) {
        this.env = env;
    }

    public void loadModel(Context context, boolean useGpu) throws Exception {
        File modelFile = ModelUtils.prepareModelFile(context, MODEL_NAME);
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        ModelUtils.configureSessionOptions(opts, useGpu, TAG);
        session = env.createSession(modelFile.getAbsolutePath(), opts);
        Log.i(TAG, "人脸检测模型加载成功, 输入: " + session.getInputNames()
                + ", 输出数: " + session.getOutputNames().size());
    }

    /**
     * 检测图像中的所有人脸
     */
    public List<DetectedFace> detect(Bitmap bitmap) throws Exception {
        if (session == null)
            throw new IllegalStateException("模型未加载");

        int origW = bitmap.getWidth();
        int origH = bitmap.getHeight();

        // 保持宽高比缩放
        float scale = Math.min((float) INPUT_SIZE / origW, (float) INPUT_SIZE / origH);
        int newW = Math.round(origW * scale);
        int newH = Math.round(origH * scale);

        Bitmap resized = Bitmap.createScaledBitmap(bitmap, newW, newH, true);

        // 创建输入张量 [1, 3, 640, 640]
        // SCRFD 预处理: BGR 通道，(pixel - 127.5) / 128.0
        float[][][][] inputData = new float[1][3][INPUT_SIZE][INPUT_SIZE];
        int[] pixels = new int[newW * newH];
        resized.getPixels(pixels, 0, newW, 0, 0, newW, newH);

        for (int y = 0; y < newH; y++) {
            for (int x = 0; x < newW; x++) {
                int pixel = pixels[y * newW + x];
                int r = (pixel >> 16) & 0xFF;
                int g = (pixel >> 8) & 0xFF;
                int b = pixel & 0xFF;
                // BGR 顺序，标准化
                inputData[0][0][y][x] = (b - 127.5f) / 128.0f;
                inputData[0][1][y][x] = (g - 127.5f) / 128.0f;
                inputData[0][2][y][x] = (r - 127.5f) / 128.0f;
            }
        }
        // Fix #3: 填充区域应为 (0 - 127.5) / 128.0 = -0.99609375，而非 0.0
        // 因为黑色像素 (0,0,0) 经过 SCRFD 预处理后应该是 -0.996
        float padVal = (0f - 127.5f) / 128.0f; // ≈ -0.996
        for (int ch = 0; ch < 3; ch++) {
            for (int y = newH; y < INPUT_SIZE; y++) {
                for (int x = 0; x < INPUT_SIZE; x++) {
                    inputData[0][ch][y][x] = padVal;
                }
            }
            for (int y = 0; y < newH; y++) {
                for (int x = newW; x < INPUT_SIZE; x++) {
                    inputData[0][ch][y][x] = padVal;
                }
            }
        }

        if (resized != bitmap)
            resized.recycle();

        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(session.getInputNames().iterator().next(), inputTensor);

        Result result = session.run(inputs);

        List<DetectedFace> faces = parseDetections(result, scale, origW, origH);
        faces = filterPlausibleFaces(faces, origW, origH);

        inputTensor.close();
        result.close();

        return nms(faces, NMS_THRESHOLD);
    }

    private List<DetectedFace> parseDetections(Result result, float scale, int origW, int origH) {
        List<DetectedFace> faces = new ArrayList<>();
        int numOutputs = (int) result.size();

        try {
            if (numOutputs == 9) {
                // 标准 SCRFD 输出: 3 个 stride × (scores, bboxes, landmarks)
                faces = parseMultiStrideOutputs(result, scale, origW, origH);
            } else if (numOutputs == 6) {
                // 无 landmarks 的 SCRFD: 3 × (scores, bboxes)
                faces = parseMultiStrideNoLandmarks(result, scale, origW, origH);
            } else {
                // 尝试通用解析
                faces = parseGenericOutput(result, scale, origW, origH);
            }
        } catch (Exception e) {
            Log.w(TAG, "标准解析失败 (outputs=" + numOutputs + "): " + e.getMessage());
            try {
                faces = parseGenericOutput(result, scale, origW, origH);
            } catch (Exception e2) {
                Log.e(TAG, "所有解析方式均失败: " + e2.getMessage());
            }
        }

        return faces;
    }

    /**
     * 标准 SCRFD 9 输出解析: stride 8/16/32 各有 scores[N,1] + bboxes[N,4] +
     * landmarks[N,10]
     */
    private List<DetectedFace> parseMultiStrideOutputs(Result result, float scale,
            int origW, int origH) throws Exception {
        List<DetectedFace> faces = new ArrayList<>();

        List<float[][]> scoreOutputs = new ArrayList<>();
        List<float[][]> bboxOutputs = new ArrayList<>();
        List<float[][]> kpsOutputs = new ArrayList<>();

        for (int i = 0; i < result.size(); i++) {
            float[][] arr = to2dArray(result.get(i).getValue());
            if (arr == null || arr.length == 0 || arr[0] == null)
                continue;
            int cols = arr[0].length;
            if (cols == 1) {
                scoreOutputs.add(arr);
            } else if (cols >= 4 && cols < 10) {
                bboxOutputs.add(arr);
            } else if (cols >= 10) {
                kpsOutputs.add(arr);
            }
        }

        boolean[] scoreUsed = new boolean[scoreOutputs.size()];
        boolean[] kpsUsed = new boolean[kpsOutputs.size()];

        for (float[][] bboxes : bboxOutputs) {
            if (bboxes == null || bboxes.length == 0)
                continue;

            int rows = bboxes.length;
            int scoreIdx = findBestRowsMatch(scoreOutputs, scoreUsed, rows);
            if (scoreIdx < 0)
                continue;

            float[][] scores = scoreOutputs.get(scoreIdx);
            scoreUsed[scoreIdx] = true;

            int kpsIdx = findBestRowsMatch(kpsOutputs, kpsUsed, rows);
            float[][] kps = null;
            if (kpsIdx >= 0) {
                kps = kpsOutputs.get(kpsIdx);
                kpsUsed[kpsIdx] = true;
            }

            decodeStrideOutput(faces, scores, bboxes, kps, scale, origW, origH);
        }

        return faces;
    }

    /**
     * 无 landmarks 的 SCRFD (6 输出)
     */
    private List<DetectedFace> parseMultiStrideNoLandmarks(Result result, float scale,
            int origW, int origH) throws Exception {
        List<DetectedFace> faces = new ArrayList<>();

        List<float[][]> scoreOutputs = new ArrayList<>();
        List<float[][]> bboxOutputs = new ArrayList<>();
        for (int i = 0; i < result.size(); i++) {
            float[][] arr = to2dArray(result.get(i).getValue());
            if (arr == null || arr.length == 0 || arr[0] == null)
                continue;
            int cols = arr[0].length;
            if (cols == 1) {
                scoreOutputs.add(arr);
            } else if (cols >= 4) {
                bboxOutputs.add(arr);
            }
        }

        boolean[] scoreUsed = new boolean[scoreOutputs.size()];
        for (float[][] bboxes : bboxOutputs) {
            if (bboxes == null || bboxes.length == 0)
                continue;
            int scoreIdx = findBestRowsMatch(scoreOutputs, scoreUsed, bboxes.length);
            if (scoreIdx < 0)
                continue;
            scoreUsed[scoreIdx] = true;
            decodeStrideOutput(faces, scoreOutputs.get(scoreIdx), bboxes, null, scale, origW, origH);
        }

        return faces;
    }

    /**
     * 通用解析（兼容合并输出格式）
     */
    private List<DetectedFace> parseGenericOutput(Result result, float scale,
            int origW, int origH) throws Exception {
        List<DetectedFace> faces = new ArrayList<>();
        float[][] output = to2dArray(result.get(0).getValue());
        if (output == null) {
            return faces;
        }

        for (float[] det : output) {
            if (det == null || det.length < 5) {
                continue;
            }

            float score = normalizeScore(det[4]);
            if (score < CONF_THRESHOLD) {
                continue;
            }

            float x1 = det[0] / scale;
            float y1 = det[1] / scale;
            float x2 = det[2] / scale;
            float y2 = det[3] / scale;

            x1 = Math.max(0, Math.min(x1, origW));
            y1 = Math.max(0, Math.min(y1, origH));
            x2 = Math.max(0, Math.min(x2, origW));
            y2 = Math.max(0, Math.min(y2, origH));

            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            DetectedFace face = new DetectedFace();
            face.box = new RectF(x1, y1, x2, y2);
            face.score = score;

            if (det.length >= 15) {
                face.landmarks = new float[5][2];
                for (int k = 0; k < 5; k++) {
                    face.landmarks[k][0] = det[5 + k * 2] / scale;
                    face.landmarks[k][1] = det[6 + k * 2] / scale;
                }
            } else {
                face.landmarks = estimateLandmarksFromBox(face.box);
            }
            faces.add(face);
        }
        return faces;
    }

    private float[][] to2dArray(Object value) {
        if (value instanceof float[][]) {
            return (float[][]) value;
        }
        if (value instanceof float[][][]) {
            float[][][] arr3 = (float[][][]) value;
            if (arr3.length == 1) {
                return arr3[0];
            }
            if (arr3.length > 0 && arr3[0] != null && arr3[0].length == 1) {
                // 兼容 [N][1][C]，压成 [N][C]
                float[][] out = new float[arr3.length][];
                for (int i = 0; i < arr3.length; i++) {
                    out[i] = arr3[i][0];
                }
                return out;
            }
        }
        return null;
    }

    private int findBestRowsMatch(List<float[][]> outputs, boolean[] used, int targetRows) {
        int bestIdx = -1;
        int bestDiff = Integer.MAX_VALUE;
        for (int i = 0; i < outputs.size(); i++) {
            if (used[i] || outputs.get(i) == null)
                continue;
            int rows = outputs.get(i).length;
            int diff = Math.abs(rows - targetRows);
            if (diff < bestDiff) {
                bestDiff = diff;
                bestIdx = i;
                if (diff == 0)
                    break;
            }
        }
        // 差异过大则不匹配
        if (bestIdx >= 0) {
            int rows = outputs.get(bestIdx).length;
            int maxAllowDiff = Math.max(16, targetRows / 4);
            if (Math.abs(rows - targetRows) > maxAllowDiff) {
                return -1;
            }
        }
        return bestIdx;
    }

    private int[] inferStrideLayout(int rows) {
        if (rows <= 0)
            return null;
        int[] anchorCandidates = { 2, 1, 3, 4 };
        for (int anchorsPerPoint : anchorCandidates) {
            if (rows % anchorsPerPoint != 0)
                continue;
            int gridTotal = rows / anchorsPerPoint;
            int grid = (int) Math.round(Math.sqrt(gridTotal));
            if (grid <= 0 || grid * grid != gridTotal)
                continue;
            if (INPUT_SIZE % grid != 0)
                continue;
            int stride = INPUT_SIZE / grid;
            if (stride <= 0)
                continue;
            return new int[] { grid, grid, anchorsPerPoint, stride, gridTotal };
        }
        return null;
    }

    private void decodeStrideOutput(List<DetectedFace> outFaces, float[][] scores, float[][] bboxes, float[][] kps,
            float scale, int origW, int origH) {
        if (scores == null || bboxes == null)
            return;

        int rows = Math.min(scores.length, bboxes.length);
        if (rows <= 0)
            return;

        int[] layout = inferStrideLayout(rows);
        int gridW;
        int gridH;
        int anchorsPerPoint;
        int stride;
        int gridTotal;
        if (layout != null) {
            gridW = layout[0];
            gridH = layout[1];
            anchorsPerPoint = Math.max(1, layout[2]);
            stride = layout[3];
            gridTotal = layout[4];
        } else {
            // 回退默认布局（尽量不越界）
            stride = 8;
            gridW = INPUT_SIZE / stride;
            gridH = INPUT_SIZE / stride;
            gridTotal = Math.max(1, gridW * gridH);
            anchorsPerPoint = Math.max(1, rows / gridTotal);
        }

        for (int i = 0; i < rows; i++) {
            if (scores[i] == null || scores[i].length < 1 || bboxes[i] == null || bboxes[i].length < 4) {
                continue;
            }

            float score = normalizeScore(scores[i][0]);
            if (score < CONF_THRESHOLD)
                continue;

            int anchorIdx = i / anchorsPerPoint;
            if (anchorIdx >= gridTotal) {
                anchorIdx = gridTotal - 1;
            }
            int gridY = anchorIdx / gridW;
            int gridX = anchorIdx % gridW;
            if (gridY < 0 || gridY >= gridH || gridX < 0 || gridX >= gridW) {
                continue;
            }

            float cx = (gridX + 0.5f) * stride;
            float cy = (gridY + 0.5f) * stride;

            float x1 = (cx - bboxes[i][0] * stride) / scale;
            float y1 = (cy - bboxes[i][1] * stride) / scale;
            float x2 = (cx + bboxes[i][2] * stride) / scale;
            float y2 = (cy + bboxes[i][3] * stride) / scale;

            x1 = Math.max(0, Math.min(x1, origW));
            y1 = Math.max(0, Math.min(y1, origH));
            x2 = Math.max(0, Math.min(x2, origW));
            y2 = Math.max(0, Math.min(y2, origH));
            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            float[][] landmarks = null;
            if (kps != null && i < kps.length && kps[i] != null && kps[i].length >= 10) {
                landmarks = new float[5][2];
                for (int k = 0; k < 5; k++) {
                    landmarks[k][0] = (cx + kps[i][k * 2] * stride) / scale;
                    landmarks[k][1] = (cy + kps[i][k * 2 + 1] * stride) / scale;
                }
            }

            DetectedFace face = new DetectedFace();
            face.box = new RectF(x1, y1, x2, y2);
            face.score = score;
            face.landmarks = (landmarks != null) ? landmarks : estimateLandmarksFromBox(face.box);
            outFaces.add(face);
        }
    }

    /**
     * 从边界框估算 5 个关键点（当模型不输出 landmarks 时的后备方案）
     */
    private float[][] estimateLandmarksFromBox(RectF box) {
        float cx = box.centerX();
        float cy = box.centerY();
        float w = box.width();
        float h = box.height();
        return new float[][] {
                { cx - w * 0.17f, cy - h * 0.12f }, // 左眼
                { cx + w * 0.17f, cy - h * 0.12f }, // 右眼
                { cx, cy + h * 0.02f }, // 鼻尖
                { cx - w * 0.14f, cy + h * 0.18f }, // 左嘴角
                { cx + w * 0.14f, cy + h * 0.18f } // 右嘴角
        };
    }

    private float normalizeScore(float raw) {
        if (Float.isNaN(raw) || Float.isInfinite(raw)) {
            return 0f;
        }
        if (raw >= 0f && raw <= 1f) {
            return raw;
        }
        double v = 1.0 / (1.0 + Math.exp(-raw));
        return (float) v;
    }

    private List<DetectedFace> filterPlausibleFaces(List<DetectedFace> faces, int origW, int origH) {
        if (faces == null || faces.isEmpty()) {
            return new ArrayList<>();
        }

        float imgArea = Math.max(1f, origW * (float) origH);
        float minSide = Math.max(MIN_FACE_SIDE_PX, Math.min(origW, origH) * MIN_FACE_SIDE_RATIO);

        List<DetectedFace> filtered = new ArrayList<>();
        for (DetectedFace face : faces) {
            if (face == null || face.box == null) {
                continue;
            }
            float w = face.box.width();
            float h = face.box.height();
            if (Float.isNaN(w) || Float.isNaN(h) || w <= 1f || h <= 1f) {
                continue;
            }

            float areaRatio = (w * h) / imgArea;
            float aspect = w / Math.max(1f, h);
            if (w < minSide || h < minSide) {
                continue;
            }
            if (areaRatio < MIN_FACE_AREA_RATIO || areaRatio > MAX_FACE_AREA_RATIO) {
                continue;
            }
            if (aspect < MIN_FACE_ASPECT || aspect > MAX_FACE_ASPECT) {
                continue;
            }

            face.score = normalizeScore(face.score);
            if (face.landmarks == null || face.landmarks.length < 5) {
                face.landmarks = estimateLandmarksFromBox(face.box);
            }
            filtered.add(face);
        }

        if (filtered.size() > MAX_OUTPUT_FACES * 4) {
            Collections.sort(filtered, (a, b) -> Float.compare(b.score, a.score));
            filtered = new ArrayList<>(filtered.subList(0, MAX_OUTPUT_FACES * 4));
        }
        return filtered;
    }

    private List<DetectedFace> nms(List<DetectedFace> faces, float threshold) {
        if (faces.isEmpty())
            return faces;
        Collections.sort(faces, (a, b) -> Float.compare(b.score, a.score));

        List<DetectedFace> result = new ArrayList<>();
        boolean[] suppressed = new boolean[faces.size()];

        for (int i = 0; i < faces.size(); i++) {
            if (suppressed[i])
                continue;
            result.add(faces.get(i));
            if (result.size() >= MAX_OUTPUT_FACES) {
                break;
            }
            for (int j = i + 1; j < faces.size(); j++) {
                if (suppressed[j])
                    continue;
                if (ModelUtils.iou(faces.get(i).box, faces.get(j).box) > threshold) {
                    suppressed[j] = true;
                }
            }
        }
        return result;
    }

    public void close() {
        try {
            if (session != null)
                session.close();
        } catch (Exception e) {
            Log.e(TAG, "关闭session失败", e);
        }
    }

    /**
     * 检测到的人脸数据
     */
    public static class DetectedFace {
        public RectF box; // 边界框
        public float score; // 置信度
        public float[][] landmarks; // 5 个关键点 [[x,y], ...]
        public float[] embedding; // 人脸特征向量（由 FaceEmbedder 填充）
    }
}