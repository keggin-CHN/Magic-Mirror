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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FaceSwapEngine {
    private static final String TAG = "FaceSwapEngine";
    private static final int MAX_DETECT_SIZE = 1920;
    private static final int TRACK_EXPIRE_FRAMES = 45;
    private final OrtEnvironment env;
    private FaceDetector detector;
    private FaceEmbedder embedder;
    private FaceSwapper swapper;
    private FaceEnhancer enhancer;
    private boolean initialized = false;
    private boolean useGpu = true;
    private boolean enableEnhancer = false;

    public interface ProgressCallback { void onProgress(String stage, int progress); }

    public FaceSwapEngine() { env = OrtEnvironment.getEnvironment(); }

    public void initialize(Context ctx, boolean useGpu, boolean enableEnhancer, ProgressCallback cb) throws Exception {
        this.useGpu = useGpu; this.enableEnhancer = enableEnhancer;
        if (cb != null) cb.onProgress("正在加载人脸检测模型...", 0);
        detector = new FaceDetector(env); detector.loadModel(ctx, useGpu);
        if (cb != null) cb.onProgress("正在加载特征提取模型...", 25);
        embedder = new FaceEmbedder(env); embedder.loadModel(ctx, useGpu);
        if (cb != null) cb.onProgress("正在加载换脸模型...", 50);
        swapper = new FaceSwapper(env); swapper.loadModel(ctx, useGpu);
        if (enableEnhancer) {
            if (cb != null) cb.onProgress("正在加载增强模型...", 75);
            enhancer = new FaceEnhancer(env); enhancer.loadModel(ctx, useGpu);
        }
        initialized = true;
        if (cb != null) cb.onProgress("模型加载完成", 100);
    }

    public boolean isInitialized() { return initialized; }

    // ==================== 人脸检测 ====================

    public List<FaceDetector.DetectedFace> detectFaces(Bitmap image) throws Exception {
        checkInit();
        Bitmap det = limitForDetection(image);
        float scale = (float) image.getWidth() / det.getWidth();
        List<FaceDetector.DetectedFace> faces = detector.detect(det);
        if (scale > 1.01f) { scaleFaces(faces, scale); if (det != image) det.recycle(); }
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
            if (exp != null) boxes.add(exp);
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
            if (fps <= 0) fps = 25f;
            int frameIndex = keyFrameMs > 0 ? Math.round(keyFrameMs / 1000f * fps) : 0;
            Bitmap frame = ret.getFrameAtTime(keyFrameMs * 1000L, MediaMetadataRetriever.OPTION_CLOSEST);
            if (frame == null) throw new RuntimeException("无法读取视频帧");
            int vw = frame.getWidth(), vh = frame.getHeight();
            List<FaceDetector.DetectedFace> faces = detectFacesInternal(frame);
            List<RectF> boxes = new ArrayList<>();
            for (FaceDetector.DetectedFace f : faces) {
                RectF exp = ModelUtils.expandSquareBox(f.box, vw, vh, 1.35f, 48);
                if (exp != null) boxes.add(exp);
            }
            boxes = ModelUtils.dedupeBoxes(boxes, 0.45f);
            frame.recycle();
            return new VideoFaceDetectionResult(boxes, vw, vh, frameIndex);
        } finally { try { ret.release(); } catch (Exception ignored) {} }
    }

    public static class VideoFaceDetectionResult {
        public List<RectF> regions; public int frameWidth, frameHeight, frameIndex;
        public VideoFaceDetectionResult(List<RectF> r, int w, int h, int fi) { regions=r; frameWidth=w; frameHeight=h; frameIndex=fi; }
    }

    // ==================== 图片换脸 ====================

    public static class FaceSourceBinding {
        public Bitmap faceImage; public RectF region; public String sourceId; public int faceIndex = -1;
        public FaceSourceBinding() {}
        public FaceSourceBinding(String id, Bitmap img, RectF r) { sourceId=id; faceImage=img; region=r; }
    }

    public Bitmap swapFace(Bitmap src, Bitmap target, ProgressCallback cb) throws Exception {
        checkInit();
        if (cb != null) cb.onProgress("检测源图人脸...", 10);
        List<FaceDetector.DetectedFace> srcFaces = detectFacesInternal(src);
        if (srcFaces.isEmpty()) throw new RuntimeException("源图中未检测到人脸");
        if (cb != null) cb.onProgress("检测目标人脸...", 20);
        List<FaceDetector.DetectedFace> tgtFaces = detectFacesInternal(target);
        if (tgtFaces.isEmpty()) throw new RuntimeException("目标图中未检测到人脸");
        if (cb != null) cb.onProgress("提取人脸特征...", 35);
        float[] emb = embedder.extractFromLandmarks(target, tgtFaces.get(0).landmarks);
        if (cb != null) cb.onProgress("执行换脸...", 50);
        Bitmap result = swapper.swapFace(src, findLargestFace(srcFaces), emb);
        if (enableEnhancer && enhancer != null && enhancer.isLoaded()) {
            if (cb != null) cb.onProgress("增强人脸质量...", 80);
            result = enhanceFaceRegion(result);
        }
        if (cb != null) cb.onProgress("完成", 100);
        return result;
    }

    /** 区域换脸 Fix #4：裁剪区域→区域内检测→区域内换脸→贴回 */
    public Bitmap swapFaceInRegions(Bitmap src, Bitmap target, List<RectF> regions, ProgressCallback cb) throws Exception {
        checkInit();
        List<FaceDetector.DetectedFace> tgtFaces = detectFacesInternal(target);
        if (tgtFaces.isEmpty()) throw new RuntimeException("目标图中未检测到人脸");
        float[] emb = embedder.extractFromLandmarks(target, tgtFaces.get(0).landmarks);
        Bitmap result = src.copy(Bitmap.Config.ARGB_8888, true);
        int cnt = 0;
        for (RectF region : regions) {
            Bitmap swapped = swapInRegion(result, region, emb);
            if (swapped != null) { result = swapped; cnt++; }
            if (cb != null) cb.onProgress("区域换脸 " + cnt + "/" + regions.size(), 50 + cnt * 40 / regions.size());
        }
        if (enableEnhancer && enhancer != null && enhancer.isLoaded() && cnt > 0) result = enhanceFaceRegion(result);
        if (cb != null) cb.onProgress("完成", 100);
        return result;
    }

    public Bitmap swapFaceInRegionsMultiSource(Bitmap src, List<FaceSourceBinding> bindings, ProgressCallback cb) throws Exception {
        checkInit();
        Map<String, float[]> embs = extractEmbeddings(bindings);
        if (embs.isEmpty()) throw new RuntimeException("所有源人脸均未检测到人脸");
        Bitmap result = src.copy(Bitmap.Config.ARGB_8888, true);
        for (FaceSourceBinding b : bindings) {
            float[] emb = embs.get(b.sourceId);
            if (emb == null || b.region == null) continue;
            Bitmap swapped = swapInRegion(result, b.region, emb);
            if (swapped != null) result = swapped;
        }
        if (enableEnhancer && enhancer != null && enhancer.isLoaded()) result = enhanceFaceRegion(result);
        if (cb != null) cb.onProgress("完成", 100);
        return result;
    }

    public Bitmap swapFaceMultiple(Bitmap src, List<FaceSourceBinding> bindings, ProgressCallback cb) throws Exception {
        checkInit();
        boolean allRegion = true;
        for (FaceSourceBinding b : bindings) { if (b.region == null) { allRegion = false; break; } }
        if (allRegion) return swapFaceInRegionsMultiSource(src, bindings, cb);
        // 无区域：全局匹配
        List<FaceDetector.DetectedFace> allFaces = detectFacesInternal(src);
        if (allFaces.isEmpty()) throw new RuntimeException("源图中未检测到人脸");
        Map<String, float[]> embs = extractEmbeddings(bindings);
        if (embs.isEmpty()) throw new RuntimeException("所有源人脸均未检测到人脸");
        Bitmap result = src.copy(Bitmap.Config.ARGB_8888, true);
        for (FaceSourceBinding b : bindings) {
            float[] emb = embs.get(b.sourceId);
            if (emb == null) continue;
            FaceDetector.DetectedFace matched = findLargestFace(allFaces);
            if (matched == null) continue;
            result = swapper.swapFace(result, matched, emb);
            allFaces = detectFacesInternal(result);
        }
        if (enableEnhancer && enhancer != null && enhancer.isLoaded()) result = enhanceFaceRegion(result);
        if (cb != null) cb.onProgress("完成", 100);
        return result;
    }

    public Bitmap swapAllFaces(Bitmap src, Bitmap target, ProgressCallback cb) throws Exception {
        checkInit();
        List<FaceDetector.DetectedFace> srcFaces = detectFacesInternal(src);
        if (srcFaces.isEmpty()) throw new RuntimeException("源图中未检测到人脸");
        List<FaceDetector.DetectedFace> tgtFaces = detectFacesInternal(target);
        if (tgtFaces.isEmpty()) throw new RuntimeException("目标图中未检测到人脸");
        float[] emb = embedder.extractFromLandmarks(target, tgtFaces.get(0).landmarks);
        Bitmap result = src.copy(Bitmap.Config.ARGB_8888, true);
        for (int i = 0; i < srcFaces.size(); i++) {
            List<FaceDetector.DetectedFace> cur = detectFacesInternal(result);
            if (cur.isEmpty()) break;
            Bitmap nr = swapper.swapFace(result, findLargestFace(cur), emb);
            if (nr != result) result.recycle();
            result = nr;
        }
        if (enableEnhancer && enhancer != null && enhancer.isLoaded()) result = enhanceFaceRegion(result);
        if (cb != null) cb.onProgress("完成", 100);
        return result;
    }

    // ==================== MainActivity 兼容接口 ====================

    public Bitmap swapFace(Bitmap src, Bitmap target, boolean useEnh, boolean swapAll, ProgressCallback cb) throws Exception {
        boolean orig = this.enableEnhancer;
        this.enableEnhancer = useEnh && enhancer != null && enhancer.isLoaded();
        try { return swapAll ? swapAllFaces(src, target, cb) : swapFace(src, target, cb); }
        finally { this.enableEnhancer = orig; }
    }

    public Bitmap swapFaceMultiSource(Bitmap src, List<FaceSourceBinding> bindings, boolean useEnh, ProgressCallback cb) throws Exception {
        boolean orig = this.enableEnhancer;
        this.enableEnhancer = useEnh && enhancer != null && enhancer.isLoaded();
        try {
            for (int i = 0; i < bindings.size(); i++) if (bindings.get(i).sourceId == null) bindings.get(i).sourceId = "src_" + i;
            return swapFaceMultiple(src, bindings, cb);
        } finally { this.enableEnhancer = orig; }
    }

    public VideoResult processVideo(Context ctx, Uri uri, Bitmap target, boolean useEnh, boolean swapAll, ProgressCallback cb) throws Exception {
        return processVideo(ctx, uri, target, useEnh, swapAll, 0, cb);
    }

    public VideoResult processVideo(Context ctx, Uri uri, Bitmap target, boolean useEnh, boolean swapAll, long keyFrameMs, ProgressCallback cb) throws Exception {
        boolean orig = this.enableEnhancer;
        this.enableEnhancer = useEnh && enhancer != null && enhancer.isLoaded();
        try {
            List<FaceDetector.DetectedFace> tf = detectFacesInternal(target);
            if (tf.isEmpty()) throw new RuntimeException("目标图中未检测到人脸");
            Map<String, float[]> embs = new HashMap<>();
            embs.put("default", embedder.extractFromLandmarks(target, tf.get(0).landmarks));
            return processVideoInternal(ctx, uri, embs, null, keyFrameMs, cb);
        } finally { this.enableEnhancer = orig; }
    }

    public VideoResult processVideoMultiSource(Context ctx, Uri uri, List<FaceSourceBinding> bindings,
                                                boolean useEnh, long keyFrameMs, ProgressCallback cb) throws Exception {
        boolean orig = this.enableEnhancer;
        this.enableEnhancer = useEnh && enhancer != null && enhancer.isLoaded();
        try {
            for (int i = 0; i < bindings.size(); i++) if (bindings.get(i).sourceId == null) bindings.get(i).sourceId = "src_" + i;
            return swapFaceVideoMultiple(ctx, uri, bindings, keyFrameMs, cb);
        } finally { this.enableEnhancer = orig; }
    }

    // ==================== 视频换脸 Fix #1,#2,#5,#6,#8 ====================

    public static class VideoResult {
        public String outputPath; public int frameCount; public float fps;
        public int width, height; public long durationMs;
        public VideoResult(String p, int fc, float f, int w, int h, long d) { outputPath=p; frameCount=fc; fps=f; width=w; height=h; durationMs=d; }
    }

    private static class FaceTrack {
        int trackId; String faceSourceId; RectF box; int missed;
        FaceTrack(int id, String sid, RectF b) { trackId=id; faceSourceId=sid; box=b; missed=0; }
    }

    public VideoResult swapFaceVideo(Context ctx, Uri uri, Bitmap target, ProgressCallback cb) throws Exception {
        checkInit();
        List<FaceDetector.DetectedFace> tf = detectFacesInternal(target);
        if (tf.isEmpty()) throw new RuntimeException("目标图中未检测到人脸");
        Map<String, float[]> embs = new HashMap<>();
        embs.put("default", embedder.extractFromLandmarks(target, tf.get(0).landmarks));
        return processVideoInternal(ctx, uri, embs, null, 0, cb);
    }

    public VideoResult swapFaceVideoMultiple(Context ctx, Uri uri, List<FaceSourceBinding> bindings,
                                              long keyFrameMs, ProgressCallback cb) throws Exception {
        checkInit();
        Map<String, float[]> embs = new HashMap<>();
        List<FaceSourceBinding> valid = new ArrayList<>();
        for (FaceSourceBinding b : bindings) {
            List<FaceDetector.DetectedFace> f = detectFacesInternal(b.faceImage);
            if (!f.isEmpty()) { embs.put(b.sourceId, embedder.extractFromLandmarks(b.faceImage, f.get(0).landmarks)); valid.add(b); }
        }
        if (embs.isEmpty()) throw new RuntimeException("所有源人脸均未检测到人脸");
        return processVideoInternal(ctx, uri, embs, valid, keyFrameMs, cb);
    }

    /** 核心视频处理：委托给 VideoProcessor（Fix #1 MediaCodec解码, Fix #2 直接编码, Fix #6 worker数） */
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
                    List<FaceDetector.DetectedFace> kfFaces = detectFacesInternal(kf);
                    buildTracksFromBindings(kfFaces, bindings, tracks);
                    kf.recycle();
                }
            } finally { try { ret.release(); } catch (Exception ignored) {} }
        }

        File outDir = new File(ctx.getCacheDir(), "swap_output"); outDir.mkdirs();
        File videoOnly = new File(outDir, "swap_v_" + System.currentTimeMillis() + ".mp4");
        File finalOut = new File(outDir, "swap_" + System.currentTimeMillis() + ".mp4");

        // Fix #6: CPU max 6, GPU 2
        int nw = useGpu ? 2 : Math.max(1, Math.min(Runtime.getRuntime().availableProcessors() - 1, 6));

        final boolean fUseTrack = useTrack;
        final Map<String, float[]> fEmbs = embs;
        final List<FaceSourceBinding> fBindings = bindings;

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
        try { VideoProcessor.copyAudioTrack(ctx, uri, videoOnly, finalOut); }
        catch (Exception e) {
            Log.w(TAG, "音频复制失败，使用无音频版本", e);
            if (!videoOnly.renameTo(finalOut)) finalOut = videoOnly;
        }

        return new VideoResult(finalOut.getAbsolutePath(), frameCount, info.fps, info.width, info.height, info.durationMs);
    }

    /** 单人视频帧处理 */
    private Bitmap processSingleFrame(Bitmap frame, float[] emb) throws Exception {
        List<FaceDetector.DetectedFace> faces = detectFacesInternal(frame);
        if (faces.isEmpty()) return frame;
        Bitmap result = swapper.swapFace(frame, findLargestFace(faces), emb);
        if (enableEnhancer && enhancer != null && enhancer.isLoaded()) result = enhanceFaceRegion(result);
        return result;
    }

    /** 多人追踪视频帧处理 Fix #5：换脸后不重新检测 */
    private Bitmap processTrackingFrame(Bitmap frame, int frameIndex, Map<String, float[]> embs,
                                         List<FaceSourceBinding> bindings, List<FaceTrack> tracks,
                                         Object trackLock) throws Exception {
        List<FaceDetector.DetectedFace> faces = detectFacesInternal(frame);
        Bitmap result = frame;

        synchronized (trackLock) {
            // 匹配检测结果到追踪（与桌面版 _match_tracks_to_detections 一致）
            matchDetectionsToTracks(faces, tracks, frame.getWidth(), frame.getHeight());

            // 对每个有匹配的追踪执行换脸
            for (FaceTrack track : tracks) {
                if (track.missed > 0) continue;
                float[] emb = embs.get(track.faceSourceId);
                if (emb == null) continue;
                // 找到匹配的人脸
                FaceDetector.DetectedFace matched = findFaceByBox(faces, track.box);
                if (matched == null) continue;
                result = swapper.swapFace(result, matched, emb);
                // Fix #5: 不重新检测！直接继续下一个追踪
            }

            // 清理过期追踪
            tracks.removeIf(t -> t.missed > TRACK_EXPIRE_FRAMES);
        }

        if (enableEnhancer && enhancer != null && enhancer.isLoaded()) result = enhanceFaceRegion(result);
        return result;
    }

    /** 多人无追踪视频帧处理 */
    private Bitmap processMultiFrame(Bitmap frame, Map<String, float[]> embs) throws Exception {
        Bitmap result = frame;
        for (float[] emb : embs.values()) {
            List<FaceDetector.DetectedFace> faces = detectFacesInternal(result);
            if (faces.isEmpty()) break;
            result = swapper.swapFace(result, findLargestFace(faces), emb);
        }
        if (enableEnhancer && enhancer != null && enhancer.isLoaded()) result = enhanceFaceRegion(result);
        return result;
    }

    // ==================== 追踪逻辑（与桌面版对齐）====================

    /** 从 bindings 建立初始追踪（与桌面版 _build_tracks_from_seed_regions 一致）*/
    private void buildTracksFromBindings(List<FaceDetector.DetectedFace> faces, List<FaceSourceBinding> bindings, List<FaceTrack> tracks) {
        int nextId = 0;
        java.util.Set<Integer> usedDet = new java.util.HashSet<>();
        for (FaceSourceBinding b : bindings) {
            if (b.region == null) continue;
            // 找到与 binding 区域 IoU 最大的人脸（阈值 0.0，与桌面版一致）
            FaceDetector.DetectedFace best = null; float bestIou = 0.0f; int bestIdx = -1;
            for (int i = 0; i < faces.size(); i++) {
                if (usedDet.contains(i)) continue;
                float iou = ModelUtils.calculateIoU(faces.get(i).box, b.region);
                if (iou > bestIou) { bestIou = iou; best = faces.get(i); bestIdx = i; }
            }
            if (best == null) {
                // IoU 失败，尝试中心距离（与桌面版一致，无阈值限制）
                float bestDist = Float.MAX_VALUE;
                for (int i = 0; i < faces.size(); i++) {
                    if (usedDet.contains(i)) continue;
                    float dist = ModelUtils.centerDistance(faces.get(i).box, b.region);
                    if (dist < bestDist) { bestDist = dist; best = faces.get(i); bestIdx = i; }
                }
            }
            if (best != null && bestIdx >= 0) {
                usedDet.add(bestIdx);
                tracks.add(new FaceTrack(nextId++, b.sourceId, new RectF(best.box)));
            }
        }
    }

    /** 匹配检测到追踪（与桌面版 _match_tracks_to_detections 一致）*/
    private void matchDetectionsToTracks(List<FaceDetector.DetectedFace> faces, List<FaceTrack> tracks, int imgW, int imgH) {
        boolean[] used = new boolean[faces.size()];

        for (FaceTrack track : tracks) {
            int bestIdx = -1; float bestIou = 0.05f;
            // 先尝试 IoU 匹配
            for (int i = 0; i < faces.size(); i++) {
                if (used[i]) continue;
                float iou = ModelUtils.calculateIoU(faces.get(i).box, track.box);
                if (iou > bestIou) { bestIou = iou; bestIdx = i; }
            }
            // IoU 失败则尝试中心距离（使用 track box 对角线 × 0.65 作为阈值，与桌面版一致）
            if (bestIdx < 0) {
                float tw = Math.max(1f, track.box.width());
                float th = Math.max(1f, track.box.height());
                float trackDiag = (float) Math.sqrt(tw * tw + th * th);
                float threshold = trackDiag * 0.65f;
                float bestDist = threshold;
                for (int i = 0; i < faces.size(); i++) {
                    if (used[i]) continue;
                    float dist = ModelUtils.centerDistance(faces.get(i).box, track.box);
                    if (dist < bestDist) { bestDist = dist; bestIdx = i; }
                }
            }
            if (bestIdx >= 0) {
                used[bestIdx] = true;
                track.box = new RectF(faces.get(bestIdx).box);
                track.missed = 0;
            } else {
                track.missed++;
            }
        }
    }

    private FaceDetector.DetectedFace findFaceByBox(List<FaceDetector.DetectedFace> faces, RectF box) {
        FaceDetector.DetectedFace best = null; float bestIou = 0;
        for (FaceDetector.DetectedFace f : faces) {
            float iou = ModelUtils.calculateIoU(f.box, box);
            if (iou > bestIou) { bestIou = iou; best = f; }
        }
        return best;
    }

    // ==================== 工具方法 ====================

    private void checkInit() { if (!initialized) throw new IllegalStateException("引擎未初始化"); }

    /** 检测人脸（内部用，自动缩放） */
    private List<FaceDetector.DetectedFace> detectFacesInternal(Bitmap image) throws Exception {
        Bitmap det = limitForDetection(image);
        float scale = (float) image.getWidth() / det.getWidth();
        List<FaceDetector.DetectedFace> faces = detector.detect(det);
        if (scale > 1.01f) { scaleFaces(faces, scale); if (det != image) det.recycle(); }
        return faces;
    }

    /** 在区域内换脸（裁剪→检测→换脸→贴回）*/
    private Bitmap swapInRegion(Bitmap image, RectF region, float[] emb) throws Exception {
        int imgW = image.getWidth(), imgH = image.getHeight();
        int x = clamp((int) region.left, 0, imgW - 1);
        int y = clamp((int) region.top, 0, imgH - 1);
        int w = clamp((int) region.width(), 1, imgW - x);
        int h = clamp((int) region.height(), 1, imgH - y);
        Bitmap crop = Bitmap.createBitmap(image, x, y, w, h);
        List<FaceDetector.DetectedFace> cropFaces = detector.detect(crop);
        if (cropFaces.isEmpty()) { crop.recycle(); return null; }
        Bitmap swappedCrop = swapper.swapFace(crop, cropFaces.get(0), emb);
        Bitmap result = image.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(result);
        canvas.drawBitmap(swappedCrop, x, y, new Paint(Paint.FILTER_BITMAP_FLAG));
        crop.recycle(); swappedCrop.recycle();
        return result;
    }

    private Map<String, float[]> extractEmbeddings(List<FaceSourceBinding> bindings) throws Exception {
        Map<String, float[]> embs = new HashMap<>();
        for (FaceSourceBinding b : bindings) {
            if (b.sourceId == null || embs.containsKey(b.sourceId)) continue;
            List<FaceDetector.DetectedFace> f = detectFacesInternal(b.faceImage);
            if (!f.isEmpty()) embs.put(b.sourceId, embedder.extractFromLandmarks(b.faceImage, f.get(0).landmarks));
        }
        return embs;
    }

    private Bitmap limitForDetection(Bitmap bmp) {
        int w = bmp.getWidth(), h = bmp.getHeight();
        if (w <= MAX_DETECT_SIZE && h <= MAX_DETECT_SIZE) return bmp;
        float scale = Math.min((float) MAX_DETECT_SIZE / w, (float) MAX_DETECT_SIZE / h);
        return Bitmap.createScaledBitmap(bmp, (int)(w * scale), (int)(h * scale), true);
    }

    private static void scaleFaces(List<FaceDetector.DetectedFace> faces, float scale) {
        for (FaceDetector.DetectedFace f : faces) {
            f.box.left *= scale; f.box.top *= scale; f.box.right *= scale; f.box.bottom *= scale;
            if (f.landmarks != null) {
                for (int i = 0; i < f.landmarks.length; i++) {
                    f.landmarks[i][0] *= scale;
                    f.landmarks[i][1] *= scale;
                }
            }
        }
    }

    private static FaceDetector.DetectedFace findLargestFace(List<FaceDetector.DetectedFace> faces) {
        if (faces == null || faces.isEmpty()) return null;
        FaceDetector.DetectedFace largest = faces.get(0);
        float maxArea = largest.box.width() * largest.box.height();
        for (int i = 1; i < faces.size(); i++) {
            float area = faces.get(i).box.width() * faces.get(i).box.height();
            if (area > maxArea) { maxArea = area; largest = faces.get(i); }
        }
        return largest;
    }

    private Bitmap enhanceFaceRegion(Bitmap image) throws Exception {
        if (enhancer == null || !enhancer.isLoaded()) return image;
        List<FaceDetector.DetectedFace> faces = detectFacesInternal(image);
        if (faces.isEmpty()) return image;
        return enhancer.enhanceAll(image, faces);
    }

    private static int clamp(int val, int min, int max) {
        return Math.max(min, Math.min(max, val));
    }

    private static int getIntMeta(MediaMetadataRetriever r, int key, int def) {
        try { String s = r.extractMetadata(key); return s != null ? Integer.parseInt(s) : def; }
        catch (Exception e) { return def; }
    }

    private static float getFloatMeta(MediaMetadataRetriever r, int key, float def) {
        try { String s = r.extractMetadata(key); return s != null ? Float.parseFloat(s) : def; }
        catch (Exception e) { return def; }
    }

    public void release() {
        try { if (detector != null) detector.close(); } catch (Exception ignored) {}
        try { if (embedder != null) embedder.close(); } catch (Exception ignored) {}
        try { if (swapper != null) swapper.close(); } catch (Exception ignored) {}
        try { if (enhancer != null) enhancer.close(); } catch (Exception ignored) {}
        initialized = false;
    }
}