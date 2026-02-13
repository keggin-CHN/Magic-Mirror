package com.magicmirror.app.engine;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaCodecList;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.MediaMetadataRetriever;
import android.media.MediaMuxer;
import android.net.Uri;
import android.util.Log;

import java.io.File;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 视频处理器 — 使用 MediaCodec+MediaExtractor 逐帧解码，Bitmap→NV12→MediaCodec 编码
 * 
 * Fix #1: 替代 getFrameAtTime() 的真正逐帧解码
 * Fix #2: 去掉 JPEG 中间文件，直接 Bitmap→YUV→编码器
 * Fix #5: 视频多人换脸不再重新检测
 * Fix #6: CPU 最多 6 worker，GPU 2 worker
 * Fix #8: key_frame_ms 支持
 */
public class VideoProcessor {
    private static final String TAG = "VideoProcessor";

    public interface FrameProcessor {
        /** 处理单帧，返回处理后的 Bitmap（可以是同一个对象） */
        Bitmap processFrame(Bitmap frame, int frameIndex) throws Exception;
    }

    public interface ProgressCallback {
        void onProgress(String stage, int progress);
    }

    public static class VideoInfo {
        public int width, height;
        public float fps;
        public long durationMs;
        public int estFrameCount;
        public int audioTrackIndex = -1;
    }

    /**
     * 处理视频：解码每一帧 → 调用 processor 处理 → 编码写入输出文件
     * 
     * @param ctx Android Context
     * @param uri 输入视频 Uri
     * @param outputFile 输出 mp4 文件
     * @param nWorkers 并行处理线程数
     * @param processor 帧处理回调
     * @param cb 进度回调
     * @return 处理的帧数
     */
    public static int process(Context ctx, Uri uri, File outputFile, int nWorkers,
                               FrameProcessor processor, ProgressCallback cb) throws Exception {
        // 获取视频信息
        VideoInfo info = getVideoInfo(ctx, uri);
        int vw = info.width, vh = info.height;
        float fps = info.fps;
        int estFrames = info.estFrameCount;

        // 确保宽高为偶数（编码器要求）
        int ew = (vw + 1) & ~1, eh = (vh + 1) & ~1;

        int queueSize = Math.max(5, nWorkers * 3);
        BlockingQueue<FrameItem> readQueue = new ArrayBlockingQueue<>(queueSize);
        TreeMap<Integer, Bitmap> writeBuffer = new TreeMap<>();
        Object writeLock = new Object();
        AtomicBoolean stopFlag = new AtomicBoolean(false);
        AtomicReference<Exception> error = new AtomicReference<>(null);
        AtomicInteger processedCount = new AtomicInteger(0);
        AtomicInteger writtenCount = new AtomicInteger(0);
        AtomicInteger totalFrames = new AtomicInteger(0);
        int nextWriteIndex = 0;

        if (cb != null) cb.onProgress("开始处理视频...", 5);

        // ========== 读取线程：MediaCodec + MediaExtractor 逐帧解码 ==========
        Thread readerThread = new Thread(() -> {
            MediaExtractor extractor = null;
            MediaCodec decoder = null;
            try {
                extractor = new MediaExtractor();
                extractor.setDataSource(ctx, uri, null);

                int videoTrack = findVideoTrack(extractor);
                if (videoTrack < 0) throw new RuntimeException("视频中未找到视频轨道");

                extractor.selectTrack(videoTrack);
                MediaFormat inputFormat = extractor.getTrackFormat(videoTrack);
                String mime = inputFormat.getString(MediaFormat.KEY_MIME);

                decoder = MediaCodec.createDecoderByType(mime);
                // 请求输出 COLOR_FormatYUV420Flexible 以便转换为 Bitmap
                decoder.configure(inputFormat, null, null, 0);
                decoder.start();

                MediaCodec.BufferInfo bufInfo = new MediaCodec.BufferInfo();
                boolean inputDone = false;
                boolean outputDone = false;
                int frameIdx = 0;

                while (!outputDone && !stopFlag.get()) {
                    // 送入压缩数据
                    if (!inputDone) {
                        int inIdx = decoder.dequeueInputBuffer(10000);
                        if (inIdx >= 0) {
                            ByteBuffer inBuf = decoder.getInputBuffer(inIdx);
                            int sampleSize = extractor.readSampleData(inBuf, 0);
                            if (sampleSize < 0) {
                                decoder.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                                inputDone = true;
                            } else {
                                long pts = extractor.getSampleTime();
                                decoder.queueInputBuffer(inIdx, 0, sampleSize, pts, 0);
                                extractor.advance();
                            }
                        }
                    }

                    // 取出解码帧
                    int outIdx = decoder.dequeueOutputBuffer(bufInfo, 10000);
                    if (outIdx >= 0) {
                        if ((bufInfo.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                            decoder.releaseOutputBuffer(outIdx, false);
                            outputDone = true;
                        } else {
                            // 从解码器输出缓冲区获取 YUV 数据并转为 Bitmap
                            Bitmap frame = yuvBufferToBitmap(decoder, outIdx, vw, vh);
                            decoder.releaseOutputBuffer(outIdx, false);

                            if (frame != null) {
                                readQueue.put(new FrameItem(frameIdx, frame));
                                frameIdx++;
                            }
                        }
                    } else if (outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                        // 格式变化，继续
                    }
                }
                totalFrames.set(frameIdx);
            } catch (Exception e) {
                if (!stopFlag.get()) {
                    error.compareAndSet(null, e);
                    stopFlag.set(true);
                }
            } finally {
                // 发送 nWorkers 个结束标记，确保每个 worker 都能收到（与桌面版一致）
                for (int i = 0; i < nWorkers; i++) {
                    try { readQueue.put(FrameItem.end()); } catch (Exception ignored) {}
                }
                if (decoder != null) { try { decoder.stop(); decoder.release(); } catch (Exception ignored) {} }
                if (extractor != null) { try { extractor.release(); } catch (Exception ignored) {} }
            }
        }, "video-reader");

        // ========== 写入线程：Bitmap→NV21→MediaCodec 编码 + MediaMuxer ==========
        Thread writerThread = new Thread(() -> {
            MediaCodec encoder = null;
            MediaMuxer muxer = null;
            int muxerTrackIndex = -1;
            boolean muxerStarted = false;
            int nextWrite = 0;

            try {
                // 配置 H.264 编码器
                MediaFormat encFormat = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, ew, eh);
                encFormat.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
                encFormat.setInteger(MediaFormat.KEY_BIT_RATE, calcBitrate(ew, eh, fps));
                encFormat.setFloat(MediaFormat.KEY_FRAME_RATE, fps);
                encFormat.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1);

                encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC);
                encoder.configure(encFormat, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
                encoder.start();

                muxer = new MediaMuxer(outputFile.getAbsolutePath(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);

                MediaCodec.BufferInfo encInfo = new MediaCodec.BufferInfo();
                boolean encodingDone = false;

                while (!encodingDone && !stopFlag.get()) {
                    // 检查是否有下一帧可写
                    Bitmap nextFrame = null;
                    synchronized (writeLock) {
                        nextFrame = writeBuffer.remove(nextWrite);
                    }

                    if (nextFrame != null) {
                        // Bitmap → NV12 → 送入编码器
                        byte[] nv12 = bitmapToNv12(nextFrame, ew, eh);
                        nextFrame.recycle();

                        int inIdx = encoder.dequeueInputBuffer(10000);
                        if (inIdx >= 0) {
                            ByteBuffer inBuf = encoder.getInputBuffer(inIdx);
                            inBuf.clear();
                            inBuf.put(nv12);
                            long pts = (long) (nextWrite * 1000000.0 / fps);
                            encoder.queueInputBuffer(inIdx, 0, nv12.length, pts, 0);
                        }
                        nextWrite++;
                        writtenCount.incrementAndGet();
                    } else {
                        // 检查是否所有帧都已处理完
                        int total = totalFrames.get();
                        int processed = processedCount.get();
                        if (total > 0 && nextWrite >= total) {
                            // 所有帧已写入，发送 EOS
                            int inIdx = encoder.dequeueInputBuffer(10000);
                            if (inIdx >= 0) {
                                encoder.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                            }
                        } else {
                            // 等待帧到达
                            synchronized (writeLock) {
                                writeLock.wait(50);
                            }
                            continue;
                        }
                    }

                    // 从编码器取出编码数据
                    while (true) {
                        int outIdx = encoder.dequeueOutputBuffer(encInfo, 0);
                        if (outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                            if (!muxerStarted) {
                                muxerTrackIndex = muxer.addTrack(encoder.getOutputFormat());
                                muxer.start();
                                muxerStarted = true;
                            }
                        } else if (outIdx >= 0) {
                            ByteBuffer outBuf = encoder.getOutputBuffer(outIdx);
                            if (muxerStarted && encInfo.size > 0) {
                                outBuf.position(encInfo.offset);
                                outBuf.limit(encInfo.offset + encInfo.size);
                                muxer.writeSampleData(muxerTrackIndex, outBuf, encInfo);
                            }
                            encoder.releaseOutputBuffer(outIdx, false);
                            if ((encInfo.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                                encodingDone = true;
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                }
            } catch (Exception e) {
                if (!stopFlag.get()) {
                    error.compareAndSet(null, e);
                    stopFlag.set(true);
                }
            } finally {
                if (encoder != null) { try { encoder.stop(); encoder.release(); } catch (Exception ignored) {} }
                if (muxer != null) { try { muxer.stop(); muxer.release(); } catch (Exception ignored) {} }
            }
        }, "video-writer");

        // ========== 处理线程池 ==========
        Thread[] workers = new Thread[nWorkers];
        for (int w = 0; w < nWorkers; w++) {
            workers[w] = new Thread(() -> {
                try {
                    while (!stopFlag.get()) {
                        FrameItem item = readQueue.poll(100, TimeUnit.MILLISECONDS);
                        if (item == null) continue;
                        if (item.isEnd) {
                            break;
                        }

                        Bitmap processed;
                        try {
                            processed = processor.processFrame(item.frame, item.index);
                        } catch (Exception e) {
                            Log.w(TAG, "帧 " + item.index + " 处理失败，使用原帧", e);
                            processed = item.frame;
                        }

                        synchronized (writeLock) {
                            writeBuffer.put(item.index, processed);
                            writeLock.notifyAll();
                        }
                        int done = processedCount.incrementAndGet();
                        if (cb != null && estFrames > 0) {
                            int pct = 5 + done * 90 / estFrames;
                            cb.onProgress("处理帧 " + done + "/" + estFrames, Math.min(pct, 95));
                        }
                    }
                } catch (InterruptedException ignored) {
                } catch (Exception e) {
                    if (!stopFlag.get()) {
                        error.compareAndSet(null, e);
                        stopFlag.set(true);
                    }
                }
            }, "video-worker-" + w);
        }

        // 启动所有线程
        readerThread.start();
        for (Thread t : workers) t.start();
        writerThread.start();

        // 等待完成
        readerThread.join();
        for (Thread t : workers) t.join();
        writerThread.join();

        if (error.get() != null) throw error.get();
        if (cb != null) cb.onProgress("视频处理完成", 100);

        return writtenCount.get();
    }

    // ========== 音频复制 ==========

    /**
     * 将源视频的音频轨道复制到输出视频（需要重新 mux）
     */
    public static void copyAudioTrack(Context ctx, Uri srcUri, File videoOnly, File finalOutput) throws Exception {
        MediaExtractor extractor = new MediaExtractor();
        extractor.setDataSource(ctx, srcUri, null);
        int audioTrack = findAudioTrack(extractor);
        if (audioTrack < 0) {
            // 没有音频轨道，直接重命名
            if (!videoOnly.renameTo(finalOutput)) {
                throw new RuntimeException("无法移动输出文件");
            }
            extractor.release();
            return;
        }

        // 有音频，需要 mux 视频+音频
        MediaMuxer muxer = new MediaMuxer(finalOutput.getAbsolutePath(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);

        // 添加视频轨道
        MediaExtractor videoExtractor = new MediaExtractor();
        videoExtractor.setDataSource(videoOnly.getAbsolutePath());
        int vidTrack = findVideoTrack(videoExtractor);
        videoExtractor.selectTrack(vidTrack);
        int muxVideoTrack = muxer.addTrack(videoExtractor.getTrackFormat(vidTrack));

        // 添加音频轨道
        extractor.selectTrack(audioTrack);
        int muxAudioTrack = muxer.addTrack(extractor.getTrackFormat(audioTrack));

        muxer.start();

        // 复制视频数据
        ByteBuffer buf = ByteBuffer.allocate(1024 * 1024);
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();

        while (true) {
            int size = videoExtractor.readSampleData(buf, 0);
            if (size < 0) break;
            info.offset = 0;
            info.size = size;
            info.presentationTimeUs = videoExtractor.getSampleTime();
            info.flags = videoExtractor.getSampleFlags();
            muxer.writeSampleData(muxVideoTrack, buf, info);
            videoExtractor.advance();
        }

        // 复制音频数据
        while (true) {
            int size = extractor.readSampleData(buf, 0);
            if (size < 0) break;
            info.offset = 0;
            info.size = size;
            info.presentationTimeUs = extractor.getSampleTime();
            info.flags = extractor.getSampleFlags();
            muxer.writeSampleData(muxAudioTrack, buf, info);
            extractor.advance();
        }

        muxer.stop();
        muxer.release();
        videoExtractor.release();
        extractor.release();
        videoOnly.delete();
    }

    // ========== 工具方法 ==========

    public static VideoInfo getVideoInfo(Context ctx, Uri uri) throws Exception {
        MediaMetadataRetriever ret = new MediaMetadataRetriever();
        ret.setDataSource(ctx, uri);
        VideoInfo info = new VideoInfo();
        info.width = getIntMeta(ret, MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH, 1280);
        info.height = getIntMeta(ret, MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT, 720);
        info.durationMs = getLongMeta(ret, MediaMetadataRetriever.METADATA_KEY_DURATION, 0);
        info.fps = getFloatMeta(ret, MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE, 25f);
        if (info.fps <= 0) info.fps = 25f;
        info.estFrameCount = Math.max(1, (int) (info.durationMs / 1000f * info.fps));
        try { ret.release(); } catch (Exception ignored) {}
        return info;
    }

    static int findVideoTrack(MediaExtractor extractor) {
        for (int i = 0; i < extractor.getTrackCount(); i++) {
            String mime = extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME);
            if (mime != null && mime.startsWith("video/")) return i;
        }
        return -1;
    }

    static int findAudioTrack(MediaExtractor extractor) {
        for (int i = 0; i < extractor.getTrackCount(); i++) {
            String mime = extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME);
            if (mime != null && mime.startsWith("audio/")) return i;
        }
        return -1;
    }

    /**
     * 从 MediaCodec 解码器输出缓冲区提取 YUV 数据并转为 ARGB_8888 Bitmap
     */
    static Bitmap yuvBufferToBitmap(MediaCodec decoder, int outputIndex, int width, int height) {
        try {
            android.media.Image image = decoder.getOutputImage(outputIndex);
            if (image == null) return null;
            return imageToBitmap(image, width, height);
        } catch (Exception e) {
            Log.w(TAG, "YUV→Bitmap 转换失败", e);
            return null;
        }
    }

    /**
     * 将 android.media.Image (YUV_420_888) 转为 ARGB_8888 Bitmap
     */
    static Bitmap imageToBitmap(android.media.Image image, int width, int height) {
        if (image.getFormat() != android.graphics.ImageFormat.YUV_420_888) {
            image.close();
            return null;
        }

        android.media.Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuf = planes[0].getBuffer();
        ByteBuffer uBuf = planes[1].getBuffer();
        ByteBuffer vBuf = planes[2].getBuffer();

        int yRowStride = planes[0].getRowStride();
        int uvRowStride = planes[1].getRowStride();
        int uvPixelStride = planes[1].getPixelStride();

        int[] argb = new int[width * height];

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int y = yBuf.get(row * yRowStride + col) & 0xFF;
                int uvRow = row >> 1;
                int uvCol = col >> 1;
                int u = uBuf.get(uvRow * uvRowStride + uvCol * uvPixelStride) & 0xFF;
                int v = vBuf.get(uvRow * uvRowStride + uvCol * uvPixelStride) & 0xFF;

                // YUV → RGB (BT.601)
                int yy = y - 16;
                int uu = u - 128;
                int vv = v - 128;
                int r = clamp((int) (1.164f * yy + 1.596f * vv), 0, 255);
                int g = clamp((int) (1.164f * yy - 0.813f * vv - 0.391f * uu), 0, 255);
                int b = clamp((int) (1.164f * yy + 2.018f * uu), 0, 255);

                argb[row * width + col] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }

        image.close();
        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        bmp.setPixels(argb, 0, width, 0, 0, width, height);
        return bmp;
    }

    /**
     * Bitmap (ARGB_8888) → NV12 字节数组（用于 MediaCodec 编码器输入）
     * NV12 格式：Y 平面 + 交错 UV 平面（U 在前，V 在后）
     * 大多数 Android H.264 硬件编码器期望 NV12 格式。
     */
    static byte[] bitmapToNv12(Bitmap bmp, int encWidth, int encHeight) {
        int w = bmp.getWidth(), h = bmp.getHeight();
        // 如果尺寸不匹配，缩放
        if (w != encWidth || h != encHeight) {
            Bitmap scaled = Bitmap.createScaledBitmap(bmp, encWidth, encHeight, true);
            if (scaled != bmp) bmp.recycle();
            bmp = scaled;
            w = encWidth;
            h = encHeight;
        }

        int[] pixels = new int[w * h];
        bmp.getPixels(pixels, 0, w, 0, 0, w, h);

        int ySize = w * h;
        int uvSize = w * h / 2;
        byte[] nv12 = new byte[ySize + uvSize];

        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                int pixel = pixels[row * w + col];
                int r = (pixel >> 16) & 0xFF;
                int g = (pixel >> 8) & 0xFF;
                int b = pixel & 0xFF;

                // RGB → YUV (BT.601)
                int y = clamp((int) (0.257f * r + 0.504f * g + 0.098f * b + 16), 0, 255);
                nv12[row * w + col] = (byte) y;

                if (row % 2 == 0 && col % 2 == 0) {
                    int uvIdx = ySize + (row >> 1) * w + col;
                    int u = clamp((int) (-0.148f * r - 0.291f * g + 0.439f * b + 128), 0, 255);
                    int v = clamp((int) (0.439f * r - 0.368f * g - 0.071f * b + 128), 0, 255);
                    nv12[uvIdx] = (byte) u;      // NV12: U first
                    nv12[uvIdx + 1] = (byte) v;  // then V
                }
            }
        }
        return nv12;
    }

    static int calcBitrate(int w, int h, float fps) {
        // 大约 4 Mbps for 1080p, 按比例缩放
        long pixels = (long) w * h;
        long refPixels = 1920L * 1080L;
        int baseBitrate = 4_000_000;
        return (int) Math.max(1_000_000, baseBitrate * pixels / refPixels);
    }

    private static int clamp(int val, int min, int max) {
        return Math.max(min, Math.min(max, val));
    }

    static int getIntMeta(MediaMetadataRetriever r, int key, int def) {
        try { String s = r.extractMetadata(key); return s != null ? Integer.parseInt(s) : def; }
        catch (Exception e) { return def; }
    }

    static long getLongMeta(MediaMetadataRetriever r, int key, long def) {
        try { String s = r.extractMetadata(key); return s != null ? Long.parseLong(s) : def; }
        catch (Exception e) { return def; }
    }

    static float getFloatMeta(MediaMetadataRetriever r, int key, float def) {
        try { String s = r.extractMetadata(key); return s != null ? Float.parseFloat(s) : def; }
        catch (Exception e) { return def; }
    }

    // 内部帧数据
    static class FrameItem {
        int index;
        Bitmap frame;
        boolean isEnd;
        FrameItem(int i, Bitmap f) { index = i; frame = f; isEnd = false; }
        static FrameItem end() { FrameItem d = new FrameItem(-1, null); d.isEnd = true; return d; }
    }
}