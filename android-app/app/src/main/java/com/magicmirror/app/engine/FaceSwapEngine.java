package com.magicmirror.app.engine;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.util.Log;
import ai.onnxruntime.OrtEnvironment;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class FaceSwapEngine {
    private static final String TAG = "FaceSwapEngine";
    private static final int MAX_DETECT_SIZE = 1920;
    private static final int TRACK_EXPIRE_FRAMES = 300;
    private static final float PSEUDO_FACE_BOX_SCALE = 0.22f;
    private static final float PSEUDO_FACE_CENTER_Y_RATIO = 0.32f;
    private static final float PSEUDO_FACE_MIN_SIDE_PX = 20f;
    private static final float PSEUDO_FACE_MAX_SIDE_PX = 220f;
    private static final float MIN_SWAP_FACE_SCORE = 0.56f;
    private static final float MIN_EMBED_FACE_SCORE = 0.52f;
    private final OrtEnvironment env;
    private FaceDetector detector;
    private FaceEmbedder embedder;
    private FaceSwapper swapper;
    private FaceEnhancer enhancer;
    private boolean initialized = false;
    private boolean useGpu = true;
    private boolean enableEnhancer = false;

    public interface ProgressCallback {
        void onProgress(String stage, int progress);
    }

    public FaceSwapEngine() {
        env = OrtEnvironment.getEnvironment();
    }

    public void initialize(Context ctx, boolean useGpu, boolean enableEnhancer, ProgressCallback cb) throws Exception {
        this.useGpu = useGpu;
        this.enableEnhancer = enableEnhancer;
        if (cb != null)
            cb.onProgress("正在加载人脸检测模型...", 0);
        detector = new FaceDetector(env);
        detector.loadModel(ctx, useGpu);
        if (cb != null)
            cb.onProgress("正在加载特征提取模型...", 25);
        embedder = new FaceEmbedder(env);
        embedder.loadModel(ctx, useGpu);
        if (cb != null)
            cb.onProgress("正在加载换脸模型...", 50);
        swapper = new FaceSwapper(env);
        swapper.loadModel(ctx, useGpu);
        if (enableEnhancer) {
            if (cb != null)
                cb.onProgress("正在加载增强模型...", 75);
            enhancer = new FaceEnhancer(env);
            enhancer.loadModel(ctx, useGpu);
        }
        initialized = true;
        if (cb != null)
            cb.onProgress("模型加载完成", 100);
    }

    public boolean isInitialized() {
        return initialized;
    }

    // ==================== 人脸检测 ====================

    public List<FaceDetector.DetectedFace> detectFaces(Bitmap image) throws Exception {
        checkInit();
        Bitmap det = limitForDetection(image);
        float scale = (float) image.getWidth() / det.getWidth();
        List<FaceDetector.DetectedFace> faces = detector.detect(det);
        if (scale > 1.01f) {
            scaleFaces(faces, scale);
            if (det != image)
                det.recycle();
        }
        return faces;
    }

    /** 检测人脸框（扩展+去重，与桌面版 _detect_face_boxes_in_frame 一致）Fix #11 */
    public List<RectF> detectFaceBoxes(Bitmap image) throws Exception {
        checkInit();
        List<FaceDetector.DetectedFace> faces = detectFaces(image);
        int imgW = image.getWidth(), imgH = image.getHeight();
        List<RectF> boxes = new ArrayList<>();
        for (FaceDetector.DetectedFace f : faces) {
            RectF exp = ModelUtils.expandSquareBox(f.box, imgW, imgH, 1.35f, 48);
            if (exp != null)
                boxes.add(exp);
        }
        return ModelUtils.dedupeBoxes(boxes, 0.45f);
    }

    /** 视频帧人脸检测 Fix #7 */
    public VideoFaceDetectionResult detectFaceBoxesInVideo(Context ctx, Uri uri, long keyFrameMs) throws Exception {
        checkInit();
        MediaMetadataRetriever ret = new MediaMetadataRetriever();
        try {
            ret.setDataSource(ctx, uri);
            float fps = VideoProcessor.getFloatMeta(ret, MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE, 25f);
            if (fps <= 0)
                fps = 25f;
            int frameIndex = keyFrameMs > 0 ? Math.round(keyFrameMs / 1000f * fps) : 0;
            Bitmap frame = ret.getFrameAtTime(keyFrameMs * 1000L, MediaMetadataRetriever.OPTION_CLOSEST);
            if (frame == null)
                throw new RuntimeException("无法读取视频帧");
            int vw = frame.getWidth(), vh = frame.getHeight();
            List<FaceDetector.DetectedFace> faces = safeDetectFaces(frame);
            List<RectF> boxes = new ArrayList<>();
            for (FaceDetector.DetectedFace f : faces) {
                RectF exp = ModelUtils.expandSquareBox(f.box, vw, vh, 1.35f, 48);
                if (exp != null)
                    boxes.add(exp);
            }
            boxes = ModelUtils.dedupeBoxes(boxes, 0.45f);
            frame.recycle();
            return new VideoFaceDetectionResult(boxes, vw, vh, frameIndex);
        } finally {
            try {
                ret.release();
            } catch (Exception ignored) {
            }
        }
    }

    public static class VideoFaceDetectionResult {
        public List<RectF> regions;
        public int frameWidth, frameHeight, frameIndex;

        public VideoFaceDetectionResult(List<RectF> r, int w, int h, int fi) {
            regions = r;
            frameWidth = w;
            frameHeight = h;
            frameIndex = fi;
        }
    }

    // ==================== 图片换脸 ====================

    public static class FaceSourceBinding {
        public Bitmap faceImage;
        public RectF region;
        public String sourceId;
        public int faceIndex = -1;

        public FaceSourceBinding() {
        }

        public FaceSourceBinding(String id, Bitmap img, RectF r) {
            sourceId = id;
            faceImage = img;
            region = r;
        }
    }

    public Bitmap swapFace(Bitmap src, Bitmap target, ProgressCallback cb) throws Exception {
        checkInit();
        if (cb != null)
            cb.onProgress("检测源图人脸...", 10);

        List<FaceDetector.DetectedFace> srcFaces = safeDetectFaces(src);
        FaceDetector.DetectedFace sourceFace = getOneFaceLikeDesktop(srcFaces);
        if (sourceFace == null || sourceFace.landmarks == null || sourceFace.landmarks.length < 5) {
            throw new RuntimeException("源图未检测到可用人脸");
        }

        if (cb != null)
            cb.onProgress("提取人脸特征...", 35);
        float[] emb = extractEmbeddingWithFallback(target);

        if (cb != null)
            cb.onProgress("执行换脸...", 50);
        Bitmap result = swapper.swapFace(src, sourceFace, emb);
        if (enableEnhancer && enhancer != null && enhancer.isLoaded()) {
            if (cb != null)
                cb.onProgress("增强人脸质量...", 80);
            result = enhanceFaceRegion(result);
        }
        if (cb != null)
            cb.onProgress("完成", 100);
        return result;
    }

    /** 区域换脸 Fix #4：裁剪区域→区域内检测→区域内换脸→贴回 */
    public Bitmap swapFaceInRegions(Bitmap src, Bitmap target, List<RectF> regions, ProgressCallback cb)
            throws Exception {
        checkInit();
        float[] emb = extractEmbeddingWithFallback(target);
        Bitmap result = src.copy(Bitmap.Config.ARGB_8888, true);
        int cnt = 0;
        for (RectF region : regions) {
            Bitmap swapped = swapInRegion(result, region, emb);
            if (swapped != null) {
                result = swapped;
                cnt++;
            }
            if (cb != null)
                cb.onProgress("区域换脸 " + cnt + "/" + regions.size(), 50 + cnt * 40 / Math.max(1, regions.size()));
        }
        if (cnt == 0) {
            Log.w(TAG, "所选区域未检测到可用人脸，按桌面版行为返回原图");
        }
        if (enableEnhancer && enhancer != null && enhancer.isLoaded())
            result = enhanceFaceRegion(result);
        if (cb != null)
            cb.onProgress("完成", 100);
        return result;
    }

    public Bitmap swapFaceInRegionsMultiSource(Bitmap src, List<FaceSourceBinding> bindings, ProgressCallback cb)
            throws Exception {
        checkInit();
        Map<String, float[]> embs = extractEmbeddings(bindings);
        if (embs.isEmpty())
            throw new RuntimeException("无法提取任何人脸特征");
        Bitmap result = src.copy(Bitmap.Config.ARGB_8888, true);
        int swappedCount = 0;
        for (FaceSourceBinding b : bindings) {
            float[] emb = embs.get(b.sourceId);
            if (emb == null || b.region == null)
                continue;
            Bitmap swapped = swapInRegion(result, b.region, emb);
            if (swapped != null) {
                result = swapped;
                swappedCount++;
            }
        }
        if (swappedCount == 0) {
            Log.w(TAG, "所有绑定区域均未检测到可用人脸，按桌面版行为返回原图");
        }
        if (enableEnhancer && enhancer != null && enhancer.isLoaded())
            result = enhanceFaceRegion(result);
        if (cb != null)
            cb.onProgress("完成", 100);
        return result;
    }

    public Bitmap swapFaceMultiple(Bitmap src, List<FaceSourceBinding> bindings, ProgressCallback cb) throws Exception {
        checkInit();
        boolean allRegion = true;
        for (FaceSourceBinding b : bindings) {
            if (b.region == null) {
                allRegion = false;
                break;
            }
        }
        if (allRegion)
            return swapFaceInRegionsMultiSource(src, bindings, cb);
        // 无区域：全局匹配
        List<FaceDetector.DetectedFace> allFaces = safeDetectFaces(src);
        Map<String, float[]> embs = extractEmbeddings(bindings);
        if (embs.isEmpty())
            throw new RuntimeException("无法提取任何人脸特征");
        Bitmap result = src.copy(Bitmap.Config.ARGB_8888, true);
        int swappedCount = 0;
        for (FaceSourceBinding b : bindings) {
            float[] emb = embs.get(b.sourceId);
            if (emb == null)
                continue;
            FaceDetector.DetectedFace matched = findBestFace(allFaces, result.getWidth(), result.getHeight());
            if (!isQualifiedForSwap(matched, result.getWidth(), result.getHeight(), MIN_SWAP_FACE_SCORE)) {
                Log.w(TAG, "多人全局模式：未找到可用目标脸，跳过 sourceId=" + b.sourceId);
                continue;
            }
            result = swapper.swapFace(result, matched, emb);
            swappedCount++;
            allFaces = safeDetectFaces(result);
        }
        if (swappedCount == 0) {
            throw new RuntimeException("未检测到可用目标脸，未执行换脸");
        }
        if (enableEnhancer && enhancer != null && enhancer.isLoaded())
            result = enhanceFaceRegion(result);
        if (cb != null)
            cb.onProgress("完成", 100);
        return result;
    }

    public Bitmap swapAllFaces(Bitmap src, Bitmap target, ProgressCallback cb) throws Exception {
        checkInit();
        List<FaceDetector.DetectedFace> srcFaces = safeDetectFaces(src);
        if (srcFaces.isEmpty()) {
            throw new RuntimeException("源图未检测到人脸，无法执行全脸换脸");
        }
        float[] emb = extractEmbeddingWithFallback(target);

        Bitmap result = src.copy(Bitmap.Config.ARGB_8888, true);
        int swappedCount = 0;
        for (int i = 0; i < srcFaces.size(); i++) {
            List<FaceDetector.DetectedFace> cur = safeDetectFaces(result);
            FaceDetector.DetectedFace face = findBestFace(cur, result.getWidth(), result.getHeight());
            if (!isQualifiedForSwap(face, result.getWidth(), result.getHeight(), MIN_SWAP_FACE_SCORE)) {
                Log.w(TAG, "全脸换脸：第 " + (i + 1) + " 次未找到可用目标脸，跳过");
                continue;
            }
            Bitmap nr = swapper.swapFace(result, face, emb);
            if (nr != result)
                result.recycle();
            result = nr;
            swappedCount++;
        }
        if (swappedCount == 0) {
            throw new RuntimeException("未检测到可用目标脸，未执行换脸");
        }
        if (enableEnhancer && enhancer != null && enhancer.isLoaded())
            result = enhanceFaceRegion(result);
        if (cb != null)
            cb.onProgress("完成", 100);
        return result;
    }

    // ==================== MainActivity 兼容接口 ====================

    public Bitmap swapFace(Bitmap src, Bitmap target, boolean useEnh, boolean swapAll, ProgressCallback cb)
            throws Exception {
        boolean orig = this.enableEnhancer;
        this.enableEnhancer = useEnh && enhancer != null && enhancer.isLoaded();
        try {
            return swapAll ? swapAllFaces(src, target, cb) : swapFace(src, target, cb);
        } finally {
            this.enableEnhancer = orig;
        }
    }

    public Bitmap swapFaceMultiSource(Bitmap src, List<FaceSourceBinding> bindings, boolean useEnh, ProgressCallback cb)
            throws Exception {
        boolean orig = this.enableEnhancer;
        this.enableEnhancer = useEnh && enhancer != null && enhancer.isLoaded();
        try {
            for (int i = 0; i < bindings.size(); i++)
                if (bindings.get(i).sourceId == null)
                    bindings.get(i).sourceId = "src_" + i;
            return swapFaceMultiple(src, bindings, cb);
        } finally {
            this.enableEnhancer = orig;
        }
    }

    public Bitmap swapFaceInRegions(Bitmap src, Bitmap target, List<RectF> regions, boolean useEnh, ProgressCallback cb)
            throws Exception {
        boolean orig = this.enableEnhancer;
        this.enableEnhancer = useEnh && enhancer != null && enhancer.isLoaded();
        try {
            return swapFaceInRegions(src, target, regions, cb);
        } finally {
            this.enableEnhancer = orig;
        }
    }

    public VideoResult processVideo(Context ctx, Uri uri, Bitmap target, boolean useEnh, boolean swapAll,
            ProgressCallback cb) throws Exception {
        return processVideo(ctx, uri, target, useEnh, swapAll, 0, cb);
    }

    public VideoResult processVideo(Context ctx, Uri uri, Bitmap target, boolean useEnh, boolean swapAll,
            long keyFrameMs, ProgressCallback cb) throws Exception {
        boolean orig = this.enableEnhancer;
        this.enableEnhancer = useEnh && enhancer != null && enhancer.isLoaded();
        try {
            Map<String, float[]> embs = new HashMap<>();
            embs.put("default", extractEmbeddingWithFallback(target));
            return processVideoInternal(ctx, uri, embs, null, keyFrameMs, cb);
        } finally {
            this.enableEnhancer = orig;
        }
    }

    public VideoResult processVideoMultiSource(Context ctx, Uri uri, List<FaceSourceBinding> bindings,
            boolean useEnh, long keyFrameMs, ProgressCallback cb) throws Exception {
        boolean orig = this.enableEnhancer;
        this.enableEnhancer = useEnh && enhancer != null && enhancer.isLoaded();
        try {
            for (int i = 0; i < bindings.size(); i++)
                if (bindings.get(i).sourceId == null)
                    bindings.get(i).sourceId = "src_" + i;
            return swapFaceVideoMultiple(ctx, uri, bindings, keyFrameMs, cb);
        } finally {
            this.enableEnhancer = orig;
        }
    }

    // ==================== 视频换脸 Fix #1,#2,#5,#6,#8 ====================

    public static class VideoResult {
        public String outputPath;
        public int frameCount;
        public float fps;
        public int width, height;
        public long durationMs;

        public VideoResult(String p, int fc, float f, int w, int h, long d) {
            outputPath = p;
            frameCount = fc;
            fps = f;
            width = w;
            height = h;
            durationMs = d;
        }
    }

    private static class FaceTrack {
        int trackId;
        String faceSourceId;
        RectF box;
        int missed;

        FaceTrack(int id, String sid, RectF b) {
            trackId = id;
            faceSourceId = sid;
            box = b;
            missed = 0;
        }
    }

    private static class TrackMatchCandidate {
        FaceTrack track;
        int detIdx;
        float iou;

        TrackMatchCandidate(FaceTrack track, int detIdx, float iou) {
            this.track = track;
            this.detIdx = detIdx;
            this.iou = iou;
        }
    }

    public VideoResult swapFaceVideo(Context ctx, Uri uri, Bitmap target, ProgressCallback cb) throws Exception {
        checkInit();
        Map<String, float[]> embs = new HashMap<>();
        embs.put("default", extractEmbeddingWithFallback(target));
        return processVideoInternal(ctx, uri, embs, null, 0, cb);
    }

    public VideoResult swapFaceVideoMultiple(Context ctx, Uri uri, List<FaceSourceBinding> bindings,
            long keyFrameMs, ProgressCallback cb) throws Exception {
        checkInit();
        Map<String, float[]> embs = new HashMap<>();
        List<FaceSourceBinding> valid = new ArrayList<>();
        for (FaceSourceBinding b : bindings) {
            if (b == null || b.faceImage == null) {
                continue;
            }
            try {
                embs.put(b.sourceId, extractEmbeddingWithFallback(b.faceImage));
                valid.add(b);
            } catch (Exception e) {
                Log.w(TAG, "跳过低质量视频人脸源 sourceId=" + b.sourceId + ": " + e.getMessage());
            }
        }
        if (embs.isEmpty())
            throw new RuntimeException("无法提取任何人脸特征");
        return processVideoInternal(ctx, uri, embs, valid, keyFrameMs, cb);
    }

    /**
     * 核心视频处理：委托给 VideoProcessor（Fix #1 MediaCodec解码, Fix #2 直接编码, Fix #6 worker数）
     */
    private VideoResult processVideoInternal(Context ctx, Uri uri, Map<String, float[]> embs,
            List<FaceSourceBinding> bindings, long keyFrameMs,
            ProgressCallback cb) throws Exception {
        VideoProcessor.VideoInfo info = VideoProcessor.getVideoInfo(ctx, uri);
        boolean singleMode = embs.size() == 1 && embs.containsKey("default");
        final float[] defEmb = singleMode ? embs.get("default") : null;
        boolean useTrack = !singleMode && bindings != null && !bindings.isEmpty();

        // 建立初始追踪（Fix #8 key_frame_ms）
        final List<FaceTrack> tracks = new ArrayList<>();
        final Object trackLock = new Object();
        if (useTrack && keyFrameMs > 0) {
            MediaMetadataRetriever ret = new MediaMetadataRetriever();
            try {
                ret.setDataSource(ctx, uri);
                Bitmap kf = ret.getFrameAtTime(keyFrameMs * 1000L, MediaMetadataRetriever.OPTION_CLOSEST);
                if (kf != null) {
                    List<FaceDetector.DetectedFace> kfFaces = safeDetectFaces(kf);
                    buildTracksFromBindings(kfFaces, bindings, tracks);
                    kf.recycle();
                }
            } finally {
                try {
                    ret.release();
                } catch (Exception ignored) {
                }
            }
        }

        if (useTrack && tracks.isEmpty()) {
            Log.w(TAG, "未建立到有效追踪，回退到多人顺序换脸模式");
            useTrack = false;
        }

        File outDir = new File(ctx.getCacheDir(), "swap_output");
        outDir.mkdirs();
        File videoOnly = new File(outDir, "swap_v_" + System.currentTimeMillis() + ".mp4");
        File finalOut = new File(outDir, "swap_" + System.currentTimeMillis() + ".mp4");

        final boolean fUseTrack = useTrack;
        final Map<String, float[]> fEmbs = embs;
        final List<FaceSourceBinding> fBindings = bindings;

        // 与桌面版 _swap_face_video_by_sources 对齐：
        // 多人追踪模式依赖 tracks 的时序更新，必须避免多 worker 乱序更新导致错配。
        int nw;
        if (fUseTrack) {
            nw = 1;
        } else {
            // 非追踪模式保持并行策略：GPU=2，CPU<=6
            nw = useGpu ? 2 : Math.max(1, Math.min(Runtime.getRuntime().availableProcessors() - 1, 6));
        }

        // 使用 VideoProcessor 处理（Fix #1 MediaCodec解码, Fix #2 直接Bitmap→YUV编码）
        int frameCount = VideoProcessor.process(ctx, uri, videoOnly, nw, (frame, frameIndex) -> {
            if (singleMode) {
                // 单人模式：检测+换脸
                return processSingleFrame(frame, defEmb);
            } else if (fUseTrack) {
                // 多人追踪模式 Fix #5：不重新检测
                return processTrackingFrame(frame, frameIndex, fEmbs, fBindings, tracks, trackLock);
            } else {
                // 多人无追踪：逐个换脸
                return processMultiFrame(frame, fEmbs);
            }
        }, cb != null ? (stage, pct) -> cb.onProgress(stage, pct) : null);

        // 复制音频轨道
        try {
            VideoProcessor.copyAudioTrack(ctx, uri, videoOnly, finalOut);
        } catch (Exception e) {
            Log.w(TAG, "音频复制失败，使用无音频版本", e);
            if (!videoOnly.renameTo(finalOut))
                finalOut = videoOnly;
        }

        return new VideoResult(finalOut.getAbsolutePath(), frameCount, info.fps, info.width, info.height,
                info.durationMs);
    }

    /** 单人视频帧处理 */
    private Bitmap processSingleFrame(Bitmap frame, float[] emb) throws Exception {
        List<FaceDetector.DetectedFace> faces = safeDetectFaces(frame);
        FaceDetector.DetectedFace face = getOneFaceLikeDesktop(faces);
        if (face == null || face.landmarks == null || face.landmarks.length < 5) {
            return frame;
        }
        Bitmap result = swapper.swapFace(frame, face, emb);
        if (enableEnhancer && enhancer != null && enhancer.isLoaded())
            result = enhanceFaceRegion(result);
        return result;
    }

    /** 多人追踪视频帧处理（与桌面版 _swap_face_video_by_sources 对齐） */
    private Bitmap processTrackingFrame(Bitmap frame, int frameIndex, Map<String, float[]> embs,
            List<FaceSourceBinding> bindings, List<FaceTrack> tracks,
            Object trackLock) throws Exception {
        List<FaceDetector.DetectedFace> faces = safeDetectFaces(frame);

        Bitmap result = frame;
        synchronized (trackLock) {
            // 匹配检测结果到追踪（与桌面版 _match_tracks_to_detections 一致）
            Map<FaceTrack, Integer> matches = matchDetectionsToTracks(faces, tracks);

            // 对每个匹配到的追踪执行换脸（不重新检测）
            for (Map.Entry<FaceTrack, Integer> entry : matches.entrySet()) {
                FaceTrack track = entry.getKey();
                if (track == null) {
                    continue;
                }
                Integer detIdxObj = entry.getValue();
                int detIdx = detIdxObj == null ? -1 : detIdxObj;
                if (detIdx < 0 || detIdx >= faces.size()) {
                    continue;
                }

                float[] emb = embs.get(track.faceSourceId);
                if (emb == null) {
                    continue;
                }

                FaceDetector.DetectedFace matched = faces.get(detIdx);
                if (matched == null || matched.landmarks == null || matched.landmarks.length < 5) {
                    continue;
                }

                result = swapper.swapFace(result, matched, emb);
            }

            // 清理过期追踪（与桌面版 missed > 300 对齐）
            tracks.removeIf(t -> t.missed > TRACK_EXPIRE_FRAMES);
        }

        if (enableEnhancer && enhancer != null && enhancer.isLoaded())
            result = enhanceFaceRegion(result);
        return result;
    }

    /** 多人无追踪视频帧处理 */
    private Bitmap processMultiFrame(Bitmap frame, Map<String, float[]> embs) throws Exception {
        Bitmap result = frame;
        for (float[] emb : embs.values()) {
            List<FaceDetector.DetectedFace> faces = safeDetectFaces(result);
            FaceDetector.DetectedFace face = getOneFaceLikeDesktop(faces);
            if (face == null || face.landmarks == null || face.landmarks.length < 5) {
                continue;
            }
            result = swapper.swapFace(result, face, emb);
        }
        if (enableEnhancer && enhancer != null && enhancer.isLoaded())
            result = enhanceFaceRegion(result);
        return result;
    }

    // ==================== 追踪逻辑（与桌面版对齐）====================

    /** 从 bindings 建立初始追踪（与桌面版 _build_tracks_from_seed_regions 一致） */
    private void buildTracksFromBindings(List<FaceDetector.DetectedFace> faces, List<FaceSourceBinding> bindings,
            List<FaceTrack> tracks) {
        int nextId = 1;
        Set<Integer> usedDet = new HashSet<>();
        List<FaceDetector.DetectedFace> dets = faces != null ? faces : new ArrayList<>();

        for (FaceSourceBinding b : bindings) {
            if (b == null || b.region == null) {
                continue;
            }

            // 先按 IoU 匹配，失败再按中心距离匹配（与桌面版一致）
            FaceDetector.DetectedFace best = null;
            int bestIdx = -1;
            float bestIou = 0.0f;

            for (int i = 0; i < dets.size(); i++) {
                if (usedDet.contains(i) || dets.get(i) == null || dets.get(i).box == null) {
                    continue;
                }
                float iou = ModelUtils.calculateIoU(dets.get(i).box, b.region);
                if (iou > bestIou) {
                    bestIou = iou;
                    best = dets.get(i);
                    bestIdx = i;
                }
            }

            if (best == null) {
                float bestDist = Float.MAX_VALUE;
                for (int i = 0; i < dets.size(); i++) {
                    if (usedDet.contains(i) || dets.get(i) == null || dets.get(i).box == null) {
                        continue;
                    }
                    float dist = ModelUtils.centerDistance(dets.get(i).box, b.region);
                    if (dist < bestDist) {
                        bestDist = dist;
                        best = dets.get(i);
                        bestIdx = i;
                    }
                }
            }

            RectF initBox;
            if (best != null && bestIdx >= 0) {
                usedDet.add(bestIdx);
                initBox = new RectF(best.box);
            } else {
                // 关键对齐点：即使关键帧未检测到人脸，也要为每个用户选区创建轨迹
                // 避免“只初始化了部分轨迹，后续只换一张脸”的问题
                initBox = new RectF(b.region);
            }

            String sid = (b.sourceId == null || b.sourceId.isEmpty()) ? ("src_" + nextId) : b.sourceId;
            tracks.add(new FaceTrack(nextId++, sid, initBox));
        }
    }

    /** 匹配检测到追踪（与桌面版 _match_tracks_to_detections 一致） */
    private Map<FaceTrack, Integer> matchDetectionsToTracks(List<FaceDetector.DetectedFace> faces,
            List<FaceTrack> tracks) {
        Map<FaceTrack, Integer> matches = new LinkedHashMap<>();
        if (tracks == null || tracks.isEmpty()) {
            return matches;
        }

        List<FaceDetector.DetectedFace> dets = faces != null ? faces : new ArrayList<>();
        if (dets.isEmpty()) {
            for (FaceTrack track : tracks) {
                track.missed++;
            }
            return matches;
        }

        // 1) 基于 IoU 的全局候选匹配（不是按 track 顺序贪心）
        List<TrackMatchCandidate> candidatePairs = new ArrayList<>();
        for (FaceTrack track : tracks) {
            if (track == null || track.box == null) {
                continue;
            }
            for (int detIdx = 0; detIdx < dets.size(); detIdx++) {
                FaceDetector.DetectedFace det = dets.get(detIdx);
                if (det == null || det.box == null) {
                    continue;
                }
                float iou = ModelUtils.calculateIoU(track.box, det.box);
                if (iou > 0.05f) {
                    candidatePairs.add(new TrackMatchCandidate(track, detIdx, iou));
                }
            }
        }
        Collections.sort(candidatePairs, (a, b) -> Float.compare(b.iou, a.iou));

        Set<FaceTrack> matchedTracks = new HashSet<>();
        Set<Integer> matchedDets = new HashSet<>();

        for (TrackMatchCandidate candidate : candidatePairs) {
            if (candidate == null || candidate.track == null) {
                continue;
            }
            if (matchedTracks.contains(candidate.track) || matchedDets.contains(candidate.detIdx)) {
                continue;
            }
            matchedTracks.add(candidate.track);
            matchedDets.add(candidate.detIdx);
            matches.put(candidate.track, candidate.detIdx);
        }

        // 2) 对未匹配轨迹做一次中心点兜底匹配（阈值=轨迹框对角线*0.65）
        for (FaceTrack track : tracks) {
            if (track == null || track.box == null || matchedTracks.contains(track)) {
                continue;
            }

            int bestIdx = -1;
            float bestDist = Float.MAX_VALUE;
            for (int detIdx = 0; detIdx < dets.size(); detIdx++) {
                if (matchedDets.contains(detIdx)) {
                    continue;
                }
                FaceDetector.DetectedFace det = dets.get(detIdx);
                if (det == null || det.box == null) {
                    continue;
                }
                float dist = ModelUtils.centerDistance(track.box, det.box);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = detIdx;
                }
            }

            if (bestIdx >= 0) {
                float tw = Math.max(1f, track.box.width());
                float th = Math.max(1f, track.box.height());
                float maxDist = (float) Math.sqrt(tw * tw + th * th) * 0.65f;
                if (bestDist <= maxDist) {
                    matchedTracks.add(track);
                    matchedDets.add(bestIdx);
                    matches.put(track, bestIdx);
                }
            }
        }

        // 3) 更新轨迹状态
        for (FaceTrack track : tracks) {
            if (track == null) {
                continue;
            }
            Integer detIdxObj = matches.get(track);
            int detIdx = detIdxObj == null ? -1 : detIdxObj;
            if (detIdx >= 0 && detIdx < dets.size() && dets.get(detIdx) != null && dets.get(detIdx).box != null) {
                track.box = new RectF(dets.get(detIdx).box);
                track.missed = 0;
            } else {
                track.missed++;
            }
        }

        return matches;
    }

    private FaceDetector.DetectedFace findFaceByBox(List<FaceDetector.DetectedFace> faces, RectF box) {
        FaceDetector.DetectedFace best = null;
        float bestIou = 0;
        for (FaceDetector.DetectedFace f : faces) {
            float iou = ModelUtils.calculateIoU(f.box, box);
            if (iou > bestIou) {
                bestIou = iou;
                best = f;
            }
        }
        return bestIou >= 0.08f ? best : null;
    }

    // ==================== 工具方法 ====================

    private void checkInit() {
        if (!initialized)
            throw new IllegalStateException("引擎未初始化");
    }

    /** 检测人脸（内部用，自动缩放） */
    private List<FaceDetector.DetectedFace> detectFacesInternal(Bitmap image) throws Exception {
        Bitmap det = limitForDetection(image);
        float scale = (float) image.getWidth() / det.getWidth();
        List<FaceDetector.DetectedFace> faces = detector.detect(det);
        if (scale > 1.01f) {
            scaleFaces(faces, scale);
            if (det != image)
                det.recycle();
        }
        return faces;
    }

    /** 在区域内换脸（裁剪→检测→换脸→贴回） */
    private Bitmap swapInRegion(Bitmap image, RectF region, float[] emb) throws Exception {
        int imgW = image.getWidth(), imgH = image.getHeight();
        int x = clamp((int) region.left, 0, imgW - 1);
        int y = clamp((int) region.top, 0, imgH - 1);
        int w = clamp((int) region.width(), 1, imgW - x);
        int h = clamp((int) region.height(), 1, imgH - y);

        Bitmap crop = Bitmap.createBitmap(image, x, y, w, h);
        List<FaceDetector.DetectedFace> cropFaces;
        try {
            cropFaces = detector.detect(crop);
        } catch (Exception e) {
            Log.w(TAG, "区域内检测失败，使用伪人脸框继续: " + e.getMessage());
            cropFaces = new ArrayList<>();
        }

        FaceDetector.DetectedFace cropFace = getOneFaceLikeDesktop(cropFaces);
        if (cropFace == null || cropFace.landmarks == null || cropFace.landmarks.length < 5) {
            crop.recycle();
            Log.w(TAG, "区域内未检测到可用人脸，跳过该区域: " + region);
            return null;
        }

        Bitmap swappedCrop = swapper.swapFace(crop, cropFace, emb);
        Bitmap result = image.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(result);
        canvas.drawBitmap(swappedCrop, x, y, new Paint(Paint.FILTER_BITMAP_FLAG));
        crop.recycle();
        swappedCrop.recycle();
        return result;
    }

    private Map<String, float[]> extractEmbeddings(List<FaceSourceBinding> bindings) throws Exception {
        Map<String, float[]> embs = new HashMap<>();
        for (FaceSourceBinding b : bindings) {
            if (b == null || b.faceImage == null || b.sourceId == null || embs.containsKey(b.sourceId))
                continue;
            try {
                embs.put(b.sourceId, extractEmbeddingWithFallback(b.faceImage));
            } catch (Exception e) {
                Log.w(TAG, "跳过低质量人脸源 sourceId=" + b.sourceId + ": " + e.getMessage());
            }
        }
        return embs;
    }

    private float[] extractEmbeddingWithFallback(Bitmap image) throws Exception {
        if (image == null) {
            throw new IllegalArgumentException("人脸特征提取输入为空");
        }

        List<FaceDetector.DetectedFace> faces = safeDetectFaces(image);
        FaceDetector.DetectedFace face = getOneFaceLikeDesktop(faces);
        if (face == null || face.landmarks == null || face.landmarks.length < 5) {
            throw new RuntimeException("目标图未检测到可用人脸");
        }
        return embedder.extractFromLandmarks(image, face.landmarks);
    }

    private List<FaceDetector.DetectedFace> safeDetectFaces(Bitmap image) {
        try {
            return detectFacesInternal(image);
        } catch (Exception e) {
            Log.w(TAG, "人脸检测失败: " + e.getMessage());
            return new ArrayList<>();
        }
    }

    private static FaceDetector.DetectedFace getOneFaceLikeDesktop(List<FaceDetector.DetectedFace> faces) {
        if (faces == null || faces.isEmpty()) {
            return null;
        }

        // 对齐桌面版 get_one_face 语义：优先返回检测列表中的第一张可用人脸
        // （检测器输出一般已按置信度排序）
        for (FaceDetector.DetectedFace f : faces) {
            if (f != null && f.box != null && f.landmarks != null && f.landmarks.length >= 5) {
                return f;
            }
        }

        // 兜底：只要有框也返回首个，后续流程再做关键点校验
        for (FaceDetector.DetectedFace f : faces) {
            if (f != null && f.box != null) {
                return f;
            }
        }
        return null;
    }

    private FaceDetector.DetectedFace createPseudoFaceForImage(Bitmap image) {
        int w = Math.max(1, image.getWidth());
        int h = Math.max(1, image.getHeight());

        float minDim = Math.min(w, h);
        float side = minDim * PSEUDO_FACE_BOX_SCALE;
        side = Math.max(PSEUDO_FACE_MIN_SIDE_PX, side);
        side = Math.min(side, PSEUDO_FACE_MAX_SIDE_PX);
        side = Math.min(side, minDim * 0.45f);

        float cx = w * 0.5f;
        float cy = h * PSEUDO_FACE_CENTER_Y_RATIO;

        float half = side * 0.5f;
        float left = cx - half;
        float top = cy - half;
        float right = cx + half;
        float bottom = cy + half;

        if (left < 0f) {
            right -= left;
            left = 0f;
        }
        if (right > w) {
            left -= (right - w);
            right = w;
        }
        if (top < 0f) {
            bottom -= top;
            top = 0f;
        }
        if (bottom > h) {
            top -= (bottom - h);
            bottom = h;
        }

        if (left < 0f)
            left = 0f;
        if (top < 0f)
            top = 0f;
        if (right > w)
            right = w;
        if (bottom > h)
            bottom = h;

        RectF box = new RectF(left, top, right, bottom);
        return createPseudoFaceForBox(box);
    }

    private FaceDetector.DetectedFace createPseudoFaceForBox(RectF box) {
        FaceDetector.DetectedFace face = new FaceDetector.DetectedFace();
        face.box = new RectF(box);
        face.score = 0f;
        face.landmarks = estimateLandmarksFromBox(face.box);
        return face;
    }

    private float[][] estimateLandmarksFromBox(RectF box) {
        float cx = box.centerX();
        float cy = box.centerY();
        float w = box.width();
        float h = box.height();
        return new float[][] {
                { cx - w * 0.17f, cy - h * 0.12f },
                { cx + w * 0.17f, cy - h * 0.12f },
                { cx, cy + h * 0.02f },
                { cx - w * 0.14f, cy + h * 0.18f },
                { cx + w * 0.14f, cy + h * 0.18f }
        };
    }

    private Bitmap limitForDetection(Bitmap bmp) {
        int w = bmp.getWidth(), h = bmp.getHeight();
        if (w <= MAX_DETECT_SIZE && h <= MAX_DETECT_SIZE)
            return bmp;
        float scale = Math.min((float) MAX_DETECT_SIZE / w, (float) MAX_DETECT_SIZE / h);
        return Bitmap.createScaledBitmap(bmp, (int) (w * scale), (int) (h * scale), true);
    }

    private static void scaleFaces(List<FaceDetector.DetectedFace> faces, float scale) {
        for (FaceDetector.DetectedFace f : faces) {
            f.box.left *= scale;
            f.box.top *= scale;
            f.box.right *= scale;
            f.box.bottom *= scale;
            if (f.landmarks != null) {
                for (int i = 0; i < f.landmarks.length; i++) {
                    f.landmarks[i][0] *= scale;
                    f.landmarks[i][1] *= scale;
                }
            }
        }
    }

    private static FaceDetector.DetectedFace findBestFace(List<FaceDetector.DetectedFace> faces, int imgW, int imgH) {
        if (faces == null || faces.isEmpty())
            return null;

        float imageArea = Math.max(1f, imgW * (float) imgH);
        FaceDetector.DetectedFace best = null;
        float bestScore = -Float.MAX_VALUE;

        for (FaceDetector.DetectedFace f : faces) {
            if (f == null || f.box == null)
                continue;

            float w = Math.max(1f, f.box.width());
            float h = Math.max(1f, f.box.height());
            float areaRatio = (w * h) / imageArea;
            float aspect = w / Math.max(1f, h);

            float detScore = normalizeScore(f.score);

            float sizeScore = clamp01((areaRatio - 0.001f) / 0.18f);
            float shapeScore = (aspect >= 0.58f && aspect <= 1.9f) ? 1f : 0.65f;
            float landmarkScore = (f.landmarks != null && f.landmarks.length >= 5) ? 1f : 0.9f;

            float score = (detScore * 0.68f + sizeScore * 0.32f) * shapeScore * landmarkScore;

            if (score > bestScore) {
                bestScore = score;
                best = f;
            }
        }

        if (best != null)
            return best;

        // 兜底：若都无效，返回第一个非空框
        for (FaceDetector.DetectedFace f : faces) {
            if (f != null && f.box != null) {
                return f;
            }
        }
        return null;
    }

    private static float normalizeScore(float raw) {
        if (Float.isNaN(raw) || Float.isInfinite(raw)) {
            return 0f;
        }
        if (raw >= 0f && raw <= 1f) {
            return raw;
        }
        return (float) (1.0 / (1.0 + Math.exp(-raw)));
    }

    private static float clamp01(float v) {
        return Math.max(0f, Math.min(1f, v));
    }

    private static boolean isQualifiedForSwap(FaceDetector.DetectedFace face, int imgW, int imgH, float minScore) {
        if (face == null || face.box == null || face.landmarks == null || face.landmarks.length < 5) {
            return false;
        }

        float w = Math.max(1f, face.box.width());
        float h = Math.max(1f, face.box.height());
        float minSide = Math.min(imgW, imgH) * 0.03f;
        if (w < minSide || h < minSide) {
            return false;
        }

        float areaRatio = (w * h) / Math.max(1f, imgW * (float) imgH);
        if (areaRatio < 0.0008f || areaRatio > 0.65f) {
            return false;
        }

        float score = normalizeScore(face.score);
        if (score < minScore) {
            return false;
        }

        float[] le = face.landmarks[0];
        float[] re = face.landmarks[1];
        float[] nose = face.landmarks[2];
        float[] lm = face.landmarks[3];
        float[] rm = face.landmarks[4];
        if (le == null || re == null || nose == null || lm == null || rm == null) {
            return false;
        }

        float eyeDx = re[0] - le[0];
        float eyeDy = re[1] - le[1];
        float eyeDist = (float) Math.sqrt(eyeDx * eyeDx + eyeDy * eyeDy);
        float eyeRatio = eyeDist / Math.max(1f, w);
        if (eyeRatio < 0.12f || eyeRatio > 0.72f) {
            return false;
        }

        float mouthY = (lm[1] + rm[1]) * 0.5f;
        if (mouthY <= nose[1]) {
            return false;
        }

        float centerX = face.box.centerX();
        float centerY = face.box.centerY();
        float yawAbs = Math.abs((le[0] + re[0]) * 0.5f - nose[0]) / Math.max(1f, w);
        float centerOffset = Math.abs(nose[0] - centerX) / Math.max(1f, w)
                + Math.abs(nose[1] - centerY) / Math.max(1f, h) * 0.5f;
        if (yawAbs > 0.22f || centerOffset > 0.55f) {
            return false;
        }

        return true;
    }

    private Bitmap enhanceFaceRegion(Bitmap image) throws Exception {
        if (enhancer == null || !enhancer.isLoaded())
            return image;
        List<FaceDetector.DetectedFace> faces = safeDetectFaces(image);
        if (faces.isEmpty())
            return image;
        return enhancer.enhanceAll(image, faces);
    }

    private static int clamp(int val, int min, int max) {
        return Math.max(min, Math.min(max, val));
    }

    private static int getIntMeta(MediaMetadataRetriever r, int key, int def) {
        try {
            String s = r.extractMetadata(key);
            return s != null ? Integer.parseInt(s) : def;
        } catch (Exception e) {
            return def;
        }
    }

    private static float getFloatMeta(MediaMetadataRetriever r, int key, float def) {
        try {
            String s = r.extractMetadata(key);
            return s != null ? Float.parseFloat(s) : def;
        } catch (Exception e) {
            return def;
        }
    }

    public void release() {
        try {
            if (detector != null)
                detector.close();
        } catch (Exception ignored) {
        }
        try {
            if (embedder != null)
                embedder.close();
        } catch (Exception ignored) {
        }
        try {
            if (swapper != null)
                swapper.close();
        } catch (Exception ignored) {
        }
        try {
            if (enhancer != null)
                enhancer.close();
        } catch (Exception ignored) {
        }
        initialized = false;
    }
}