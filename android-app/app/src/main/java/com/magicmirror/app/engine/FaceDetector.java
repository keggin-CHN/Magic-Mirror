package com.magicmirror.app.engine;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;

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

    private OrtSession session;
    private final OrtEnvironment env;

    public FaceDetector(OrtEnvironment env) {
        this.env = env;
    }

    public void loadModel(Context context, boolean useGpu) throws Exception {
        byte[] modelBytes = ModelUtils.loadModel(context, MODEL_NAME);
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        ModelUtils.configureSessionOptions(opts, useGpu, TAG);
        session = env.createSession(modelBytes, opts);
        Log.i(TAG, "人脸检测模型加载成功, 输入: " + session.getInputNames()
                + ", 输出数: " + session.getOutputNames().size());
    }

    /**
     * 检测图像中的所有人脸
     */
    public List<DetectedFace> detect(Bitmap bitmap) throws Exception {
        if (session == null) throw new IllegalStateException("模型未加载");

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
        float padVal = (0f - 127.5f) / 128.0f;  // ≈ -0.996
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

        if (resized != bitmap) resized.recycle();

        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(session.getInputNames().iterator().next(), inputTensor);

        Result result = session.run(inputs);

        List<DetectedFace> faces = parseDetections(result, scale, origW, origH);

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
     * 标准 SCRFD 9 输出解析: stride 8/16/32 各有 scores[N,1] + bboxes[N,4] + landmarks[N,10]
     */
    private List<DetectedFace> parseMultiStrideOutputs(Result result, float scale,
                                                        int origW, int origH) throws Exception {
        List<DetectedFace> faces = new ArrayList<>();
        int[] strides = {8, 16, 32};

        for (int s = 0; s < 3; s++) {
            int stride = strides[s];
            int gridH = INPUT_SIZE / stride;
            int gridW = INPUT_SIZE / stride;

            float[][] scores = (float[][]) result.get(s * 3).getValue();
            float[][] bboxes = (float[][]) result.get(s * 3 + 1).getValue();
            float[][] kps = (float[][]) result.get(s * 3 + 2).getValue();

            int numAnchors = scores.length;
            // SCRFD 2.5g 每个位置 2 个 anchor
            int gridTotal = gridH * gridW;
            int anchorsPerPoint = (gridTotal > 0) ? Math.max(1, numAnchors / gridTotal) : 1;

            for (int i = 0; i < numAnchors; i++) {
                float score = scores[i][0];
                if (score < CONF_THRESHOLD) continue;

                int anchorIdx = i / anchorsPerPoint;
                int gridY = anchorIdx / gridW;
                int gridX = anchorIdx % gridW;

                float cx = (gridX + 0.5f) * stride;
                float cy = (gridY + 0.5f) * stride;

                // 解码 bbox: distance from anchor center
                float x1 = (cx - bboxes[i][0] * stride) / scale;
                float y1 = (cy - bboxes[i][1] * stride) / scale;
                float x2 = (cx + bboxes[i][2] * stride) / scale;
                float y2 = (cy + bboxes[i][3] * stride) / scale;

                x1 = Math.max(0, Math.min(x1, origW));
                y1 = Math.max(0, Math.min(y1, origH));
                x2 = Math.max(0, Math.min(x2, origW));
                y2 = Math.max(0, Math.min(y2, origH));

                // 解码 landmarks
                float[][] landmarks = new float[5][2];
                if (kps[i].length >= 10) {
                    for (int k = 0; k < 5; k++) {
                        landmarks[k][0] = (cx + kps[i][k * 2] * stride) / scale;
                        landmarks[k][1] = (cy + kps[i][k * 2 + 1] * stride) / scale;
                    }
                }

                DetectedFace face = new DetectedFace();
                face.box = new RectF(x1, y1, x2, y2);
                face.score = score;
                face.landmarks = landmarks;
                faces.add(face);
            }
        }
        return faces;
    }

    /**
     * 无 landmarks 的 SCRFD (6 输出)
     */
    private List<DetectedFace> parseMultiStrideNoLandmarks(Result result, float scale,
                                                            int origW, int origH) throws Exception {
        List<DetectedFace> faces = new ArrayList<>();
        int[] strides = {8, 16, 32};

        for (int s = 0; s < 3; s++) {
            int stride = strides[s];
            int gridW = INPUT_SIZE / stride;

            float[][] scores = (float[][]) result.get(s * 2).getValue();
            float[][] bboxes = (float[][]) result.get(s * 2 + 1).getValue();

            int numAnchors = scores.length;
            int gridTotal = (INPUT_SIZE / stride) * (INPUT_SIZE / stride);
            int anchorsPerPoint = (gridTotal > 0) ? Math.max(1, numAnchors / gridTotal) : 1;

            for (int i = 0; i < numAnchors; i++) {
                float score = scores[i][0];
                if (score < CONF_THRESHOLD) continue;

                int anchorIdx = i / anchorsPerPoint;
                int gridY = anchorIdx / gridW;
                int gridX = anchorIdx % gridW;

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

                DetectedFace face = new DetectedFace();
                face.box = new RectF(x1, y1, x2, y2);
                face.score = score;
                // 无 landmarks 时用框中心估算
                face.landmarks = estimateLandmarksFromBox(face.box);
                faces.add(face);
            }
        }
        return faces;
    }

    /**
     * 通用解析（兼容合并输出格式）
     */
    private List<DetectedFace> parseGenericOutput(Result result, float scale,
                                                   int origW, int origH) throws Exception {
        List<DetectedFace> faces = new ArrayList<>();
        float[][] output = (float[][]) result.get(0).getValue();

        for (float[] det : output) {
            if (det.length >= 5 && det[4] >= CONF_THRESHOLD) {
                float x1 = det[0] / scale;
                float y1 = det[1] / scale;
                float x2 = det[2] / scale;
                float y2 = det[3] / scale;

                x1 = Math.max(0, Math.min(x1, origW));
                y1 = Math.max(0, Math.min(y1, origH));
                x2 = Math.max(0, Math.min(x2, origW));
                y2 = Math.max(0, Math.min(y2, origH));

                DetectedFace face = new DetectedFace();
                face.box = new RectF(x1, y1, x2, y2);
                face.score = det[4];

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
        }
        return faces;
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
            {cx - w * 0.17f, cy - h * 0.12f},  // 左眼
            {cx + w * 0.17f, cy - h * 0.12f},  // 右眼
            {cx,             cy + h * 0.02f},   // 鼻尖
            {cx - w * 0.14f, cy + h * 0.18f},  // 左嘴角
            {cx + w * 0.14f, cy + h * 0.18f}   // 右嘴角
        };
    }

    private List<DetectedFace> nms(List<DetectedFace> faces, float threshold) {
        if (faces.isEmpty()) return faces;
        Collections.sort(faces, (a, b) -> Float.compare(b.score, a.score));

        List<DetectedFace> result = new ArrayList<>();
        boolean[] suppressed = new boolean[faces.size()];

        for (int i = 0; i < faces.size(); i++) {
            if (suppressed[i]) continue;
            result.add(faces.get(i));
            for (int j = i + 1; j < faces.size(); j++) {
                if (suppressed[j]) continue;
                if (ModelUtils.iou(faces.get(i).box, faces.get(j).box) > threshold) {
                    suppressed[j] = true;
                }
            }
        }
        return result;
    }

    public void close() {
        try {
            if (session != null) session.close();
        } catch (Exception e) {
            Log.e(TAG, "关闭session失败", e);
        }
    }

    /**
     * 检测到的人脸数据
     */
    public static class DetectedFace {
        public RectF box;           // 边界框
        public float score;         // 置信度
        public float[][] landmarks; // 5 个关键点 [[x,y], ...]
        public float[] embedding;   // 人脸特征向量（由 FaceEmbedder 填充）
    }
}