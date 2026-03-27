package com.magicmirror.app.engine;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.util.HashMap;
import java.util.Map;

/**
 * ArcFace 人脸特征提取器 — 使用 arcface_w600k_r50.onnx 模型
 * 输入: 对齐后的 112x112 人脸图像，BGR 通道，归一化到 [-1, 1]
 * 输出: 512 维特征向量（L2 归一化）
 */
public class FaceEmbedder {
    private static final String TAG = "FaceEmbedder";
    private static final String MODEL_NAME = "arcface_w600k_r50.onnx";
    private static final int INPUT_SIZE = 112;

    private OrtSession session;
    private final OrtEnvironment env;

    public FaceEmbedder(OrtEnvironment env) {
        this.env = env;
    }

    public void loadModel(Context context, boolean useGpu) throws Exception {
        byte[] modelBytes = ModelUtils.loadModel(context, MODEL_NAME);
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        ModelUtils.configureSessionOptions(opts, useGpu, TAG);
        session = env.createSession(modelBytes, opts);
        Log.i(TAG, "人脸特征提取模型加载成功");
    }

    /**
     * 提取人脸特征向量
     * @param alignedFace 对齐后的 112x112 人脸图像
     * @return 512 维 L2 归一化特征向量
     */
    public float[] extractEmbedding(Bitmap alignedFace) throws Exception {
        if (session == null) throw new IllegalStateException("模型未加载");

        // ArcFace 预处理: BGR 通道，(pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1
        // 等价于归一化到 [-1, 1]
        float[][][][] inputData = ModelUtils.bitmapToBgrNormalized(
                alignedFace, INPUT_SIZE,
                new float[]{127.5f, 127.5f, 127.5f},  // mean (BGR)
                new float[]{127.5f, 127.5f, 127.5f}   // std (BGR)
        );

        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(session.getInputNames().iterator().next(), inputTensor);

        OrtSession.Result result = session.run(inputs);

        // 输出: [1, 512]
        float[][] output = (float[][]) result.get(0).getValue();
        float[] embedding = output[0];

        inputTensor.close();
        result.close();

        return ModelUtils.l2Normalize(embedding);
    }

    /**
     * 从原图和关键点直接提取特征（对齐 → 提取）
     */
    public float[] extractFromLandmarks(Bitmap source, float[][] landmarks) throws Exception {
        Bitmap aligned = ModelUtils.alignFace(source, landmarks, INPUT_SIZE);
        float[] embedding = extractEmbedding(aligned);
        aligned.recycle();
        return embedding;
    }

    public void close() {
        try {
            if (session != null) session.close();
        } catch (Exception e) {
            Log.e(TAG, "关闭session失败", e);
        }
    }
}