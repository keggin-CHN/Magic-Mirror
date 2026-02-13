package com.magicmirror.app;

import android.Manifest;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.SeekBar;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.content.ContextCompat;

import com.magicmirror.app.engine.FaceDetector;
import com.magicmirror.app.engine.FaceSwapEngine;
import com.magicmirror.app.engine.ModelUtils;
import com.magicmirror.app.view.FaceOverlayView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    private enum Mode { IMAGE, VIDEO }
    private Mode currentMode = Mode.IMAGE;

    // 通用控件
    private ProgressBar progressBar;
    private TextView tvStatus;
    private Switch switchGpu, switchEnhancer, switchSwapAll;
    private Button btnModeImage, btnModeVideo;

    // 图片模式控件
    private LinearLayout layoutImageMode;
    private ImageView ivSource, ivTarget, ivResult;
    private FaceOverlayView faceOverlay;
    private Button btnSelectSource, btnSelectTarget, btnSwap, btnSave;
    private Button btnRegionSelect, btnClearRegions, btnAddFaceSource;
    private LinearLayout layoutMultiFaceControls, layoutFaceBindings;
    private TextView tvResultLabel, tvFaceHint;
    private CardView cardResult;

    // 视频模式控件
    private LinearLayout layoutVideoMode;
    private ImageView ivVideoThumb, ivVideoTarget, ivKeyFrameThumb;
    private FaceOverlayView videoFaceOverlay;
    private Button btnSelectVideo, btnSelectVideoTarget, btnSwapVideo;
    private Button btnDetectVideoFaces, btnAddVideoFaceSource;
    private LinearLayout layoutKeyFrame, layoutVideoFaceControls, layoutVideoFaceBindings;
    private SeekBar seekbarKeyFrame;
    private TextView tvVideoInfo, tvVideoResultInfo, tvKeyFrameTime;
    private CardView cardVideoResult;
    private long videoDurationMs = 0;
    private long videoKeyFrameMs = 0;
    private final List<FaceSourceEntry> videoFaceSourceEntries = new ArrayList<>();
    private List<RectF> videoDetectedBoxes = new ArrayList<>();

    // 数据
    private Bitmap sourceBitmap, targetBitmap, resultBitmap;
    private Bitmap videoTargetBitmap;
    private Uri selectedVideoUri;
    private List<FaceDetector.DetectedFace> detectedFaces = new ArrayList<>();
    private final List<FaceSourceEntry> faceSourceEntries = new ArrayList<>();
    private boolean regionSelectMode = false;

    private FaceSwapEngine engine;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private boolean engineInitialized = false;
    private boolean isProcessing = false;

    // 多人脸源条目
    private static class FaceSourceEntry {
        String id;
        Bitmap faceImage;
        RectF region; // 绑定到源图的哪个区域
        int bindingIndex; // 绑定到第几个检测到的人脸 (-1 = 用区域)
        View itemView;
        FaceSourceEntry(String id) { this.id = id; bindingIndex = -1; }
    }

    // 图片选择器
    private final ActivityResultLauncher<Intent> sourceImagePicker =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Uri uri = result.getData().getData();
                    if (uri != null) loadSourceImage(uri);
                }
            });

    private final ActivityResultLauncher<Intent> targetImagePicker =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Uri uri = result.getData().getData();
                    if (uri != null) loadTargetImage(uri);
                }
            });

    private final ActivityResultLauncher<Intent> videoPicker =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Uri uri = result.getData().getData();
                    if (uri != null) loadVideo(uri);
                }
            });

    private final ActivityResultLauncher<Intent> videoTargetPicker =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Uri uri = result.getData().getData();
                    if (uri != null) loadVideoTarget(uri);
                }
            });

    // 多人脸源图片选择器（动态）
    private ActivityResultLauncher<Intent> faceSourcePicker;
    private ActivityResultLauncher<Intent> videoFaceSourcePicker;
    private int pendingFaceSourceIndex = -1;
    private int pendingVideoFaceSourceIndex = -1;

    private final ActivityResultLauncher<String[]> permissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestMultiplePermissions(), permissions -> {
                boolean allGranted = true;
                for (Boolean granted : permissions.values()) {
                    if (!granted) allGranted = false;
                }
                if (allGranted) {
                    checkModelsAndInit();
                } else {
                    Toast.makeText(this, R.string.permission_required, Toast.LENGTH_LONG).show();
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initViews();
        checkPermissions();
    }

    private void initViews() {
        // 通用
        progressBar = findViewById(R.id.progress_bar);
        tvStatus = findViewById(R.id.tv_status);
        switchGpu = findViewById(R.id.switch_gpu);
        switchEnhancer = findViewById(R.id.switch_enhancer);
        switchSwapAll = findViewById(R.id.switch_swap_all);
        btnModeImage = findViewById(R.id.btn_mode_image);
        btnModeVideo = findViewById(R.id.btn_mode_video);

        // 图片模式
        layoutImageMode = findViewById(R.id.layout_image_mode);
        ivSource = findViewById(R.id.iv_source);
        ivTarget = findViewById(R.id.iv_target);
        ivResult = findViewById(R.id.iv_result);
        faceOverlay = findViewById(R.id.face_overlay);
        btnSelectSource = findViewById(R.id.btn_select_source);
        btnSelectTarget = findViewById(R.id.btn_select_target);
        btnSwap = findViewById(R.id.btn_swap);
        btnSave = findViewById(R.id.btn_save);
        tvResultLabel = findViewById(R.id.tv_result_label);
        cardResult = findViewById(R.id.card_result);
        btnRegionSelect = findViewById(R.id.btn_region_select);
        btnClearRegions = findViewById(R.id.btn_clear_regions);
        btnAddFaceSource = findViewById(R.id.btn_add_face_source);
        layoutMultiFaceControls = findViewById(R.id.layout_multi_face_controls);
        layoutFaceBindings = findViewById(R.id.layout_face_bindings);
        tvFaceHint = findViewById(R.id.tv_face_hint);

        // 视频模式
        layoutVideoMode = findViewById(R.id.layout_video_mode);
        ivVideoThumb = findViewById(R.id.iv_video_thumb);
        ivVideoTarget = findViewById(R.id.iv_video_target);
        ivKeyFrameThumb = findViewById(R.id.iv_key_frame_thumb);
        videoFaceOverlay = findViewById(R.id.video_face_overlay);
        btnSelectVideo = findViewById(R.id.btn_select_video);
        btnSelectVideoTarget = findViewById(R.id.btn_select_video_target);
        btnSwapVideo = findViewById(R.id.btn_swap_video);
        btnDetectVideoFaces = findViewById(R.id.btn_detect_video_faces);
        btnAddVideoFaceSource = findViewById(R.id.btn_add_video_face_source);
        layoutKeyFrame = findViewById(R.id.layout_key_frame);
        layoutVideoFaceControls = findViewById(R.id.layout_video_face_controls);
        layoutVideoFaceBindings = findViewById(R.id.layout_video_face_bindings);
        seekbarKeyFrame = findViewById(R.id.seekbar_key_frame);
        tvVideoInfo = findViewById(R.id.tv_video_info);
        tvVideoResultInfo = findViewById(R.id.tv_video_result_info);
        tvKeyFrameTime = findViewById(R.id.tv_key_frame_time);
        cardVideoResult = findViewById(R.id.card_video_result);

        // 默认值
        switchGpu.setChecked(true);
        switchEnhancer.setChecked(false);
        switchSwapAll.setChecked(false);

        // 模式切换
        btnModeImage.setOnClickListener(v -> switchMode(Mode.IMAGE));
        btnModeVideo.setOnClickListener(v -> switchMode(Mode.VIDEO));

        // 图片模式按钮
        btnSelectSource.setOnClickListener(v -> pickImage(sourceImagePicker));
        btnSelectTarget.setOnClickListener(v -> pickImage(targetImagePicker));
        btnSwap.setOnClickListener(v -> performImageSwap());
        btnSave.setOnClickListener(v -> saveResult());

        // 区域选择
        btnRegionSelect.setOnClickListener(v -> toggleRegionSelect());
        btnClearRegions.setOnClickListener(v -> {
            faceOverlay.clearSelectedRegions();
            faceOverlay.clearHighlights();
        });
        btnAddFaceSource.setOnClickListener(v -> addFaceSourceEntry());

        // 人脸点击回调
        faceOverlay.setOnFaceClickedListener((index, box) -> {
            if (!faceSourceEntries.isEmpty()) {
                // 绑定到最后一个未绑定的 entry
                for (FaceSourceEntry e : faceSourceEntries) {
                    if (e.bindingIndex < 0 && e.region == null) {
                        e.bindingIndex = index;
                        e.region = box;
                        faceOverlay.highlightFace(index);
                        updateFaceBindingUI(e);
                        break;
                    }
                }
            }
        });

        // 区域选择回调
        faceOverlay.setOnRegionSelectedListener(region -> {
            for (FaceSourceEntry e : faceSourceEntries) {
                if (e.region == null && e.bindingIndex < 0) {
                    e.region = region;
                    updateFaceBindingUI(e);
                    break;
                }
            }
        });

        // 多人脸源图片选择器
        faceSourcePicker = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(), result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null && pendingFaceSourceIndex >= 0) {
                        Uri uri = result.getData().getData();
                        if (uri != null && pendingFaceSourceIndex < faceSourceEntries.size()) {
                            loadFaceSourceImage(uri, faceSourceEntries.get(pendingFaceSourceIndex));
                        }
                    }
                    pendingFaceSourceIndex = -1;
                });

        // 视频多人脸源图片选择器
        videoFaceSourcePicker = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(), result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null && pendingVideoFaceSourceIndex >= 0) {
                        Uri uri = result.getData().getData();
                        if (uri != null && pendingVideoFaceSourceIndex < videoFaceSourceEntries.size()) {
                            loadVideoFaceSourceImage(uri, videoFaceSourceEntries.get(pendingVideoFaceSourceIndex));
                        }
                    }
                    pendingVideoFaceSourceIndex = -1;
                });

        // 视频人脸点击回调
        videoFaceOverlay.setOnFaceClickedListener((index, box) -> {
            if (!videoFaceSourceEntries.isEmpty()) {
                for (FaceSourceEntry e : videoFaceSourceEntries) {
                    if (e.bindingIndex < 0 && e.region == null) {
                        e.bindingIndex = index;
                        e.region = box;
                        videoFaceOverlay.highlightFace(index);
                        updateVideoFaceBindingUI(e);
                        break;
                    }
                }
            }
        });

        // 视频模式按钮
        btnSelectVideo.setOnClickListener(v -> pickVideo());
        btnSelectVideoTarget.setOnClickListener(v -> pickImage(videoTargetPicker));
        btnSwapVideo.setOnClickListener(v -> performVideoSwap());
        btnDetectVideoFaces.setOnClickListener(v -> detectVideoFaces());
        btnAddVideoFaceSource.setOnClickListener(v -> addVideoFaceSourceEntry());

        // 关键帧 SeekBar
        seekbarKeyFrame.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar sb, int progress, boolean fromUser) {
                if (fromUser && videoDurationMs > 0) {
                    videoKeyFrameMs = (long) progress * videoDurationMs / sb.getMax();
                    tvKeyFrameTime.setText(String.format("%.1fs", videoKeyFrameMs / 1000.0));
                }
            }
            @Override public void onStartTrackingTouch(SeekBar sb) {}
            @Override
            public void onStopTrackingTouch(SeekBar sb) {
                // 拖拽结束时更新关键帧缩略图
                updateKeyFrameThumb();
            }
        });

        btnSwap.setEnabled(false);
        btnSwapVideo.setEnabled(false);

        // swapAll 开关变化时显示/隐藏多人脸控件
        switchSwapAll.setOnCheckedChangeListener((btn, checked) -> {
            layoutMultiFaceControls.setVisibility(checked && sourceBitmap != null ? View.VISIBLE : View.GONE);
            tvFaceHint.setVisibility(checked && sourceBitmap != null ? View.VISIBLE : View.GONE);
        });
    }

    // ==================== 模式切换 ====================

    private void switchMode(Mode mode) {
        if (isProcessing) {
            Toast.makeText(this, R.string.status_processing, Toast.LENGTH_SHORT).show();
            return;
        }
        currentMode = mode;
        if (mode == Mode.IMAGE) {
            layoutImageMode.setVisibility(View.VISIBLE);
            layoutVideoMode.setVisibility(View.GONE);
            btnModeImage.setBackgroundTintList(android.content.res.ColorStateList.valueOf(0xFFE94560));
            btnModeImage.setTextColor(0xFFFFFFFF);
            btnModeVideo.setBackgroundTintList(android.content.res.ColorStateList.valueOf(0xFF333355));
            btnModeVideo.setTextColor(0xFFAAAACC);
        } else {
            layoutImageMode.setVisibility(View.GONE);
            layoutVideoMode.setVisibility(View.VISIBLE);
            btnModeVideo.setBackgroundTintList(android.content.res.ColorStateList.valueOf(0xFFE94560));
            btnModeVideo.setTextColor(0xFFFFFFFF);
            btnModeImage.setBackgroundTintList(android.content.res.ColorStateList.valueOf(0xFF333355));
            btnModeImage.setTextColor(0xFFAAAACC);
        }
    }

    // ==================== 权限和引擎 ====================

    private void checkPermissions() {
        if (android.os.Build.VERSION.SDK_INT >= 33) {
            boolean hasImages = ContextCompat.checkSelfPermission(this,
                    Manifest.permission.READ_MEDIA_IMAGES) == PackageManager.PERMISSION_GRANTED;
            boolean hasVideo = ContextCompat.checkSelfPermission(this,
                    Manifest.permission.READ_MEDIA_VIDEO) == PackageManager.PERMISSION_GRANTED;
            if (hasImages && hasVideo) {
                checkModelsAndInit();
            } else {
                permissionLauncher.launch(new String[]{
                        Manifest.permission.READ_MEDIA_IMAGES,
                        Manifest.permission.READ_MEDIA_VIDEO
                });
            }
        } else {
            if (ContextCompat.checkSelfPermission(this,
                    Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                checkModelsAndInit();
            } else {
                permissionLauncher.launch(new String[]{
                        Manifest.permission.READ_EXTERNAL_STORAGE,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE
                });
            }
        }
    }

    // ==================== 模型下载检查 ====================

    private static final String[] REQUIRED_MODELS = {
        "scrfd_2.5g.onnx", "arcface_w600k_r50.onnx", "inswapper_128_fp16.onnx"
    };
    private static final String[] OPTIONAL_MODELS = { "gfpgan_1.4.onnx" };
    private static final String MODEL_BASE_URL = "https://huggingface.co/magicmirror/models/resolve/main/";

    private void checkModelsAndInit() {
        // 检查模型是否存在
        List<String> missing = new ArrayList<>();
        for (String m : REQUIRED_MODELS) {
            if (!modelExists(m)) missing.add(m);
        }
        if (switchEnhancer.isChecked()) {
            for (String m : OPTIONAL_MODELS) {
                if (!modelExists(m)) missing.add(m);
            }
        }

        if (missing.isEmpty()) {
            initEngine();
        } else {
            // 提示下载
            showModelDownloadDialog(missing);
        }
    }

    private boolean modelExists(String name) {
        // 检查外部存储
        File ext = new File(getExternalFilesDir("models"), name);
        if (ext.exists() && ext.length() > 0) return true;
        // 检查 assets
        try {
            InputStream is = getAssets().open("models/" + name);
            is.close();
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private void showModelDownloadDialog(List<String> missing) {
        StringBuilder sb = new StringBuilder();
        sb.append(getString(R.string.model_check_assets)).append("\n\n");
        for (String m : missing) sb.append("• ").append(m).append("\n");

        new AlertDialog.Builder(this)
            .setTitle(R.string.model_download_title)
            .setMessage(sb.toString())
            .setPositiveButton("下载", (d, w) -> downloadModels(missing))
            .setNegativeButton("跳过", (d, w) -> initEngine())
            .setCancelable(false)
            .show();
    }

    private void downloadModels(List<String> models) {
        setStatus(getString(R.string.status_initializing));
        progressBar.setVisibility(View.VISIBLE);
        progressBar.setProgress(0);

        executor.execute(() -> {
            try {
                File dir = getExternalFilesDir("models");
                if (dir != null && !dir.exists()) dir.mkdirs();

                for (int i = 0; i < models.size(); i++) {
                    String name = models.get(i);
                    int fi = i;
                    mainHandler.post(() -> setStatus(getString(R.string.model_download_progress, name, 0)));

                    URL url = new URL(MODEL_BASE_URL + name);
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    conn.setConnectTimeout(30000);
                    conn.setReadTimeout(60000);
                    conn.connect();

                    int total = conn.getContentLength();
                    File outFile = new File(dir, name);
                    try (InputStream in = conn.getInputStream();
                         FileOutputStream fos = new FileOutputStream(outFile)) {
                        byte[] buf = new byte[8192];
                        int read, downloaded = 0;
                        while ((read = in.read(buf)) != -1) {
                            fos.write(buf, 0, read);
                            downloaded += read;
                            int pct = total > 0 ? (int)(100L * downloaded / total) : 0;
                            int overallPct = (int)((fi * 100f + pct) / models.size());
                            mainHandler.post(() -> {
                                setStatus(getString(R.string.model_download_progress, name, pct));
                                progressBar.setProgress(overallPct);
                            });
                        }
                    }
                    conn.disconnect();
                }

                mainHandler.post(() -> {
                    setStatus(getString(R.string.model_download_done));
                    initEngine();
                });
            } catch (Exception e) {
                Log.e(TAG, "模型下载失败", e);
                mainHandler.post(() -> {
                    progressBar.setVisibility(View.GONE);
                    setStatus(getString(R.string.model_download_failed, e.getMessage()));
                    // 仍然尝试初始化
                    initEngine();
                });
            }
        });
    }

    private void initEngine() {
        engine = new FaceSwapEngine();
        setStatus(getString(R.string.status_initializing));
        progressBar.setVisibility(View.VISIBLE);

        executor.execute(() -> {
            try {
                engine.initialize(this, switchGpu.isChecked(), switchEnhancer.isChecked(),
                        (stage, progress) -> mainHandler.post(() -> {
                            setStatus(stage);
                            progressBar.setProgress(progress);
                        }));

                mainHandler.post(() -> {
                    engineInitialized = true;
                    progressBar.setVisibility(View.GONE);
                    setStatus(getString(R.string.status_models_ready));
                    updateButtons();
                });
            } catch (Exception e) {
                Log.e(TAG, "引擎初始化失败", e);
                mainHandler.post(() -> {
                    progressBar.setVisibility(View.GONE);
                    setStatus(getString(R.string.status_models_failed, e.getMessage()));
                    Toast.makeText(this, R.string.model_check_assets, Toast.LENGTH_LONG).show();
                });
            }
        });
    }

    // ==================== 图片选择 ====================

    private void pickImage(ActivityResultLauncher<Intent> launcher) {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        intent.setType("image/*");
        launcher.launch(intent);
    }

    private void pickVideo() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Video.Media.EXTERNAL_CONTENT_URI);
        intent.setType("video/*");
        videoPicker.launch(intent);
    }

    private void loadSourceImage(Uri uri) {
        try {
            if (sourceBitmap != null && !sourceBitmap.isRecycled()) sourceBitmap.recycle();
            sourceBitmap = ModelUtils.loadBitmapRobust(this, uri);
            sourceBitmap = limitBitmapSize(sourceBitmap, 1920);
            ivSource.setImageBitmap(sourceBitmap);
            setStatus(getString(R.string.status_source_selected, sourceBitmap.getWidth(), sourceBitmap.getHeight()));
            clearImageResult();

            // 设置 overlay 尺寸
            faceOverlay.setImageSize(sourceBitmap.getWidth(), sourceBitmap.getHeight());
            faceOverlay.clearAll();

            // 自动检测人脸
            if (engineInitialized) {
                autoDetectFaces();
            }

            // 显示多人脸控件
            if (switchSwapAll.isChecked()) {
                layoutMultiFaceControls.setVisibility(View.VISIBLE);
                tvFaceHint.setVisibility(View.VISIBLE);
            }

            updateButtons();
        } catch (Exception e) {
            Toast.makeText(this, getString(R.string.load_image_failed, e.getMessage()), Toast.LENGTH_SHORT).show();
        }
    }

    private void autoDetectFaces() {
        if (sourceBitmap == null || !engineInitialized) return;
        setStatus(getString(R.string.status_detecting_faces));
        executor.execute(() -> {
            try {
                List<FaceDetector.DetectedFace> faces = engine.detectFaces(sourceBitmap);
                mainHandler.post(() -> {
                    detectedFaces = faces;
                    List<RectF> boxes = new ArrayList<>();
                    for (FaceDetector.DetectedFace f : faces) boxes.add(f.box);
                    faceOverlay.setFaceBoxes(boxes);
                    if (faces.isEmpty()) {
                        setStatus(getString(R.string.status_no_faces));
                    } else {
                        setStatus(getString(R.string.status_faces_detected, faces.size()));
                    }
                });
            } catch (Exception e) {
                Log.w(TAG, "自动检测失败: " + e.getMessage());
                mainHandler.post(() -> setStatus(getString(R.string.status_no_faces)));
            }
        });
    }

    private void loadTargetImage(Uri uri) {
        try {
            if (targetBitmap != null && !targetBitmap.isRecycled()) targetBitmap.recycle();
            targetBitmap = ModelUtils.loadBitmapRobust(this, uri);
            targetBitmap = limitBitmapSize(targetBitmap, 1920);
            ivTarget.setImageBitmap(targetBitmap);
            setStatus(getString(R.string.status_target_selected));
            clearImageResult();
            updateButtons();
        } catch (Exception e) {
            Toast.makeText(this, getString(R.string.load_image_failed, e.getMessage()), Toast.LENGTH_SHORT).show();
        }
    }

    private void loadVideo(Uri uri) {
        try {
            selectedVideoUri = uri;
            MediaMetadataRetriever retriever = new MediaMetadataRetriever();
            retriever.setDataSource(this, uri);

            Bitmap thumb = retriever.getFrameAtTime(0);
            if (thumb != null) ivVideoThumb.setImageBitmap(thumb);

            String w = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH);
            String h = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT);
            String dur = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
            retriever.release();

            videoDurationMs = dur != null ? Long.parseLong(dur) : 0;
            videoKeyFrameMs = 0;
            String info = (w != null ? w : "?") + "x" + (h != null ? h : "?")
                    + " · " + String.format("%.1f", videoDurationMs / 1000.0) + "s";
            tvVideoInfo.setText(info);
            tvVideoInfo.setVisibility(View.VISIBLE);

            // 显示关键帧选择器和人脸检测控件
            layoutKeyFrame.setVisibility(View.VISIBLE);
            layoutVideoFaceControls.setVisibility(View.VISIBLE);
            seekbarKeyFrame.setProgress(0);
            tvKeyFrameTime.setText("0.0s");
            if (thumb != null) {
                ivKeyFrameThumb.setImageBitmap(thumb);
            }
            videoFaceOverlay.clearAll();
            videoDetectedBoxes.clear();
            clearVideoFaceSourceEntries();

            setStatus(getString(R.string.status_video_selected));
            cardVideoResult.setVisibility(View.GONE);
            updateButtons();
        } catch (Exception e) {
            Toast.makeText(this, getString(R.string.load_video_failed, e.getMessage()), Toast.LENGTH_SHORT).show();
        }
    }

    private void updateKeyFrameThumb() {
        if (selectedVideoUri == null) return;
        executor.execute(() -> {
            try {
                MediaMetadataRetriever ret = new MediaMetadataRetriever();
                ret.setDataSource(this, selectedVideoUri);
                Bitmap frame = ret.getFrameAtTime(videoKeyFrameMs * 1000L, MediaMetadataRetriever.OPTION_CLOSEST);
                ret.release();
                if (frame != null) {
                    mainHandler.post(() -> ivKeyFrameThumb.setImageBitmap(frame));
                }
            } catch (Exception e) {
                Log.w(TAG, "获取关键帧缩略图失败", e);
            }
        });
    }

    private void detectVideoFaces() {
        if (selectedVideoUri == null || !engineInitialized || isProcessing) return;
        setProcessing(true);
        setStatus(getString(R.string.status_detecting_faces));

        executor.execute(() -> {
            try {
                FaceSwapEngine.VideoFaceDetectionResult result =
                        engine.detectFaceBoxesInVideo(this, selectedVideoUri, videoKeyFrameMs);
                mainHandler.post(() -> {
                    videoDetectedBoxes = result.regions;
                    videoFaceOverlay.setImageSize(result.frameWidth, result.frameHeight);
                    videoFaceOverlay.setFaceBoxes(result.regions);
                    if (result.regions.isEmpty()) {
                        setStatus(getString(R.string.status_no_faces));
                    } else {
                        setStatus(getString(R.string.video_faces_detected, result.regions.size()));
                    }
                    setProcessing(false);
                });
            } catch (Exception e) {
                Log.e(TAG, "视频人脸检测失败", e);
                mainHandler.post(() -> {
                    setStatus(getString(R.string.swap_failed, e.getMessage()));
                    setProcessing(false);
                });
            }
        });
    }

    private void loadVideoTarget(Uri uri) {
        try {
            if (videoTargetBitmap != null && !videoTargetBitmap.isRecycled()) videoTargetBitmap.recycle();
            videoTargetBitmap = ModelUtils.loadBitmapRobust(this, uri);
            videoTargetBitmap = limitBitmapSize(videoTargetBitmap, 1920);
            ivVideoTarget.setImageBitmap(videoTargetBitmap);
            setStatus(getString(R.string.status_video_target_selected));
            cardVideoResult.setVisibility(View.GONE);
            updateButtons();
        } catch (Exception e) {
            Toast.makeText(this, getString(R.string.load_image_failed, e.getMessage()), Toast.LENGTH_SHORT).show();
        }
    }

    // ==================== 区域选择 ====================

    private void toggleRegionSelect() {
        regionSelectMode = !regionSelectMode;
        faceOverlay.setRegionSelectMode(regionSelectMode);
        btnRegionSelect.setText(regionSelectMode ? R.string.btn_region_done : R.string.btn_region_select);
        btnRegionSelect.setBackgroundTintList(android.content.res.ColorStateList.valueOf(
                regionSelectMode ? 0xFFE94560 : 0xFF533483));
    }

    // ==================== 多人脸源绑定 ====================

    private void addFaceSourceEntry() {
        int idx = faceSourceEntries.size();
        FaceSourceEntry entry = new FaceSourceEntry("face_" + idx);

        // 创建 UI 行
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, 4, 0, 4);

        TextView label = new TextView(this);
        label.setText(getString(R.string.face_source_binding, idx + 1));
        label.setTextColor(0xFFCCCCDD);
        label.setTextSize(12);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
        label.setLayoutParams(lp);

        ImageView thumb = new ImageView(this);
        thumb.setLayoutParams(new LinearLayout.LayoutParams(48, 48));
        thumb.setScaleType(ImageView.ScaleType.CENTER_CROP);
        thumb.setBackgroundColor(0xFF16213E);

        Button btnPick = new Button(this);
        btnPick.setText(R.string.btn_select_target);
        btnPick.setTextSize(10);
        btnPick.setBackgroundTintList(android.content.res.ColorStateList.valueOf(0xFF0F3460));
        btnPick.setTextColor(0xFFFFFFFF);
        LinearLayout.LayoutParams bp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, 32);
        bp.setMarginStart(8);
        btnPick.setLayoutParams(bp);
        btnPick.setOnClickListener(v -> {
            pendingFaceSourceIndex = faceSourceEntries.indexOf(entry);
            pickImage(faceSourcePicker);
        });

        Button btnRemove = new Button(this);
        btnRemove.setText(R.string.btn_remove_binding);
        btnRemove.setTextSize(10);
        btnRemove.setBackgroundTintList(android.content.res.ColorStateList.valueOf(0xFF333355));
        btnRemove.setTextColor(0xFFAAAACC);
        LinearLayout.LayoutParams rp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, 32);
        rp.setMarginStart(4);
        btnRemove.setLayoutParams(rp);
        btnRemove.setOnClickListener(v -> removeFaceSourceEntry(entry));

        row.addView(label);
        row.addView(thumb);
        row.addView(btnPick);
        row.addView(btnRemove);

        entry.itemView = row;
        faceSourceEntries.add(entry);
        layoutFaceBindings.addView(row);
    }

    private void removeFaceSourceEntry(FaceSourceEntry entry) {
        faceSourceEntries.remove(entry);
        layoutFaceBindings.removeView(entry.itemView);
        if (entry.faceImage != null && !entry.faceImage.isRecycled()) entry.faceImage.recycle();
        faceOverlay.clearHighlights();
        // 重新高亮剩余绑定
        for (FaceSourceEntry e : faceSourceEntries) {
            if (e.bindingIndex >= 0) faceOverlay.highlightFace(e.bindingIndex);
        }
    }

    private void updateFaceBindingUI(FaceSourceEntry entry) {
        if (entry.itemView instanceof LinearLayout) {
            LinearLayout row = (LinearLayout) entry.itemView;
            TextView label = (TextView) row.getChildAt(0);
            int idx = faceSourceEntries.indexOf(entry) + 1;
            String bindInfo;
            if (entry.bindingIndex >= 0) {
                bindInfo = getString(R.string.face_source_binding, idx) + " → 人脸#" + (entry.bindingIndex + 1);
            } else if (entry.region != null) {
                bindInfo = getString(R.string.face_source_binding, idx) + " → 区域";
            } else {
                bindInfo = getString(R.string.face_source_binding, idx) + " → 未绑定";
            }
            label.setText(bindInfo);
        }
    }

    private void loadFaceSourceImage(Uri uri, FaceSourceEntry entry) {
        try {
            if (entry.faceImage != null && !entry.faceImage.isRecycled()) entry.faceImage.recycle();
            entry.faceImage = ModelUtils.loadBitmapRobust(this, uri);
            entry.faceImage = limitBitmapSize(entry.faceImage, 1920);

            // 更新缩略图
            if (entry.itemView instanceof LinearLayout) {
                ImageView thumb = (ImageView) ((LinearLayout) entry.itemView).getChildAt(1);
                thumb.setImageBitmap(entry.faceImage);
            }
            updateFaceBindingUI(entry);
        } catch (Exception e) {
            Toast.makeText(this, getString(R.string.load_image_failed, e.getMessage()), Toast.LENGTH_SHORT).show();
        }
    }

    // ==================== 换脸执行 ====================

    private void performImageSwap() {
        if (sourceBitmap == null || targetBitmap == null || !engineInitialized || isProcessing) return;
        setProcessing(true);
        setStatus(getString(R.string.status_processing));
        progressBar.setVisibility(View.VISIBLE);
        progressBar.setProgress(0);

        boolean useEnhancer = switchEnhancer.isChecked();
        boolean swapAll = switchSwapAll.isChecked();

        // 构建多人脸源绑定
        List<FaceSwapEngine.FaceSourceBinding> bindings = new ArrayList<>();
        if (swapAll && !faceSourceEntries.isEmpty()) {
            for (FaceSourceEntry e : faceSourceEntries) {
                if (e.faceImage != null) {
                    FaceSwapEngine.FaceSourceBinding b = new FaceSwapEngine.FaceSourceBinding();
                    b.faceImage = e.faceImage;
                    b.region = e.region;
                    b.faceIndex = e.bindingIndex;
                    bindings.add(b);
                }
            }
        }

        final List<FaceSwapEngine.FaceSourceBinding> finalBindings = bindings;
        executor.execute(() -> {
            try {
                Bitmap result;
                if (swapAll && !finalBindings.isEmpty()) {
                    result = engine.swapFaceMultiSource(sourceBitmap, finalBindings, useEnhancer,
                            (stage, progress) -> mainHandler.post(() -> {
                                setStatus(stage);
                                progressBar.setProgress(progress);
                            }));
                } else {
                    result = engine.swapFace(sourceBitmap, targetBitmap, useEnhancer, swapAll,
                            (stage, progress) -> mainHandler.post(() -> {
                                setStatus(stage);
                                progressBar.setProgress(progress);
                            }));
                }

                mainHandler.post(() -> {
                    if (result != null) {
                        if (resultBitmap != null && !resultBitmap.isRecycled()) resultBitmap.recycle();
                        resultBitmap = result;
                        ivResult.setImageBitmap(resultBitmap);
                        cardResult.setVisibility(View.VISIBLE);
                        tvResultLabel.setVisibility(View.VISIBLE);
                        btnSave.setVisibility(View.VISIBLE);
                        setStatus(getString(R.string.status_swap_done));
                    } else {
                        setStatus(getString(R.string.swap_failed, "结果为空"));
                    }
                    setProcessing(false);
                    progressBar.setVisibility(View.GONE);
                });
            } catch (Exception e) {
                Log.e(TAG, "换脸失败", e);
                mainHandler.post(() -> {
                    setStatus(getString(R.string.swap_failed, e.getMessage()));
                    setProcessing(false);
                    progressBar.setVisibility(View.GONE);
                });
            }
        });
    }

    private void performVideoSwap() {
        if (selectedVideoUri == null || !engineInitialized || isProcessing) return;

        boolean useEnhancer = switchEnhancer.isChecked();
        boolean swapAll = switchSwapAll.isChecked();

        // 判断是否使用多人脸源模式
        boolean useMultiSource = !videoFaceSourceEntries.isEmpty();
        List<FaceSwapEngine.FaceSourceBinding> bindings = new ArrayList<>();
        if (useMultiSource) {
            for (FaceSourceEntry e : videoFaceSourceEntries) {
                if (e.faceImage != null) {
                    FaceSwapEngine.FaceSourceBinding b = new FaceSwapEngine.FaceSourceBinding();
                    b.faceImage = e.faceImage;
                    b.region = e.region;
                    b.faceIndex = e.bindingIndex;
                    bindings.add(b);
                }
            }
        }

        // 多人脸源模式不需要 videoTargetBitmap，单人模式需要
        if (!useMultiSource && videoTargetBitmap == null) return;

        setProcessing(true);
        setStatus(getString(R.string.status_processing));
        progressBar.setVisibility(View.VISIBLE);
        progressBar.setProgress(0);

        final List<FaceSwapEngine.FaceSourceBinding> finalBindings = bindings;
        final boolean fUseMulti = useMultiSource;

        executor.execute(() -> {
            try {
                FaceSwapEngine.VideoResult vr;
                if (fUseMulti && !finalBindings.isEmpty()) {
                    vr = engine.processVideoMultiSource(
                            this, selectedVideoUri, finalBindings, useEnhancer, videoKeyFrameMs,
                            (stage, progress) -> mainHandler.post(() -> {
                                setStatus(stage);
                                progressBar.setProgress(progress);
                            }));
                } else {
                    vr = engine.processVideo(
                            this, selectedVideoUri, videoTargetBitmap, useEnhancer, swapAll, videoKeyFrameMs,
                            (stage, progress) -> mainHandler.post(() -> {
                                setStatus(stage);
                                progressBar.setProgress(progress);
                            }));
                }

                mainHandler.post(() -> {
                    if (vr != null && vr.outputPath != null) {
                        cardVideoResult.setVisibility(View.VISIBLE);
                        tvVideoResultInfo.setText(getString(R.string.video_swap_done, vr.frameCount) + "\n" + vr.outputPath);
                        setStatus(getString(R.string.video_swap_done, vr.frameCount));
                    } else {
                        setStatus(getString(R.string.video_swap_failed, "结果为空"));
                    }
                    setProcessing(false);
                    progressBar.setVisibility(View.GONE);
                });
            } catch (Exception e) {
                Log.e(TAG, "视频换脸失败", e);
                mainHandler.post(() -> {
                    setStatus(getString(R.string.video_swap_failed, e.getMessage()));
                    setProcessing(false);
                    progressBar.setVisibility(View.GONE);
                });
            }
        });
    }

    // ==================== 视频多人脸源绑定 ====================

    private void addVideoFaceSourceEntry() {
        int idx = videoFaceSourceEntries.size();
        FaceSourceEntry entry = new FaceSourceEntry("vface_" + idx);

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, 4, 0, 4);

        TextView label = new TextView(this);
        label.setText(getString(R.string.face_source_binding, idx + 1));
        label.setTextColor(0xFFCCCCDD);
        label.setTextSize(12);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
        label.setLayoutParams(lp);

        ImageView thumb = new ImageView(this);
        thumb.setLayoutParams(new LinearLayout.LayoutParams(48, 48));
        thumb.setScaleType(ImageView.ScaleType.CENTER_CROP);
        thumb.setBackgroundColor(0xFF16213E);

        Button btnPick = new Button(this);
        btnPick.setText(R.string.btn_select_target);
        btnPick.setTextSize(10);
        btnPick.setBackgroundTintList(android.content.res.ColorStateList.valueOf(0xFF0F3460));
        btnPick.setTextColor(0xFFFFFFFF);
        LinearLayout.LayoutParams bp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, 32);
        bp.setMarginStart(8);
        btnPick.setLayoutParams(bp);
        btnPick.setOnClickListener(v -> {
            pendingVideoFaceSourceIndex = videoFaceSourceEntries.indexOf(entry);
            pickImage(videoFaceSourcePicker);
        });

        Button btnRemove = new Button(this);
        btnRemove.setText(R.string.btn_remove_binding);
        btnRemove.setTextSize(10);
        btnRemove.setBackgroundTintList(android.content.res.ColorStateList.valueOf(0xFF333355));
        btnRemove.setTextColor(0xFFAAAACC);
        LinearLayout.LayoutParams rp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, 32);
        rp.setMarginStart(4);
        btnRemove.setLayoutParams(rp);
        btnRemove.setOnClickListener(v -> removeVideoFaceSourceEntry(entry));

        row.addView(label);
        row.addView(thumb);
        row.addView(btnPick);
        row.addView(btnRemove);

        entry.itemView = row;
        videoFaceSourceEntries.add(entry);
        layoutVideoFaceBindings.addView(row);
    }

    private void removeVideoFaceSourceEntry(FaceSourceEntry entry) {
        videoFaceSourceEntries.remove(entry);
        layoutVideoFaceBindings.removeView(entry.itemView);
        if (entry.faceImage != null && !entry.faceImage.isRecycled()) entry.faceImage.recycle();
        videoFaceOverlay.clearHighlights();
        for (FaceSourceEntry e : videoFaceSourceEntries) {
            if (e.bindingIndex >= 0) videoFaceOverlay.highlightFace(e.bindingIndex);
        }
    }

    private void clearVideoFaceSourceEntries() {
        for (FaceSourceEntry e : videoFaceSourceEntries) {
            if (e.faceImage != null && !e.faceImage.isRecycled()) e.faceImage.recycle();
        }
        videoFaceSourceEntries.clear();
        layoutVideoFaceBindings.removeAllViews();
    }

    private void updateVideoFaceBindingUI(FaceSourceEntry entry) {
        if (entry.itemView instanceof LinearLayout) {
            LinearLayout row = (LinearLayout) entry.itemView;
            TextView label = (TextView) row.getChildAt(0);
            int idx = videoFaceSourceEntries.indexOf(entry) + 1;
            String bindInfo;
            if (entry.bindingIndex >= 0) {
                bindInfo = getString(R.string.face_source_binding, idx) + " → 人脸#" + (entry.bindingIndex + 1);
            } else if (entry.region != null) {
                bindInfo = getString(R.string.face_source_binding, idx) + " → 区域";
            } else {
                bindInfo = getString(R.string.face_source_binding, idx) + " → 未绑定";
            }
            label.setText(bindInfo);
        }
    }

    private void loadVideoFaceSourceImage(Uri uri, FaceSourceEntry entry) {
        try {
            if (entry.faceImage != null && !entry.faceImage.isRecycled()) entry.faceImage.recycle();
            entry.faceImage = ModelUtils.loadBitmapRobust(this, uri);
            entry.faceImage = limitBitmapSize(entry.faceImage, 1920);
            if (entry.itemView instanceof LinearLayout) {
                ImageView thumb = (ImageView) ((LinearLayout) entry.itemView).getChildAt(1);
                thumb.setImageBitmap(entry.faceImage);
            }
            updateVideoFaceBindingUI(entry);
        } catch (Exception e) {
            Toast.makeText(this, getString(R.string.load_image_failed, e.getMessage()), Toast.LENGTH_SHORT).show();
        }
    }

    // ==================== 保存结果 ====================

    private void saveResult() {
        if (resultBitmap == null) return;
        try {
            ContentValues values = new ContentValues();
            values.put(MediaStore.Images.Media.DISPLAY_NAME, "MagicMirror_" + System.currentTimeMillis() + ".png");
            values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
            if (android.os.Build.VERSION.SDK_INT >= 29) {
                values.put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/MagicMirror");
            }

            Uri uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
            if (uri != null) {
                OutputStream os = getContentResolver().openOutputStream(uri);
                if (os != null) {
                    resultBitmap.compress(Bitmap.CompressFormat.PNG, 100, os);
                    os.close();
                }
                Toast.makeText(this, R.string.save_success, Toast.LENGTH_SHORT).show();
                setStatus(getString(R.string.save_success));
            }
        } catch (Exception e) {
            Toast.makeText(this, getString(R.string.save_failed, e.getMessage()), Toast.LENGTH_SHORT).show();
        }
    }

    // ==================== 辅助方法 ====================

    private void clearImageResult() {
        if (resultBitmap != null && !resultBitmap.isRecycled()) resultBitmap.recycle();
        resultBitmap = null;
        ivResult.setImageResource(0);
        cardResult.setVisibility(View.GONE);
        tvResultLabel.setVisibility(View.GONE);
    }

    private void updateButtons() {
        boolean hasSource = sourceBitmap != null;
        boolean hasTarget = targetBitmap != null;
        boolean hasVideo = selectedVideoUri != null;
        boolean hasVideoTarget = videoTargetBitmap != null;

        btnSwap.setEnabled(hasSource && hasTarget && engineInitialized && !isProcessing);
        btnSave.setEnabled(resultBitmap != null);
        boolean hasVideoFaceSources = !videoFaceSourceEntries.isEmpty();
        btnSwapVideo.setEnabled(hasVideo && (hasVideoTarget || hasVideoFaceSources) && engineInitialized && !isProcessing);
        btnDetectVideoFaces.setEnabled(hasVideo && engineInitialized && !isProcessing);
    }

    private void setProcessing(boolean processing) {
        isProcessing = processing;
        updateButtons();
        btnSelectSource.setEnabled(!processing);
        btnSelectTarget.setEnabled(!processing);
        btnSelectVideo.setEnabled(!processing);
        btnSelectVideoTarget.setEnabled(!processing);
        switchGpu.setEnabled(!processing);
        switchEnhancer.setEnabled(!processing);
        switchSwapAll.setEnabled(!processing);
    }

    private void setStatus(String text) {
        tvStatus.setText(text);
    }

    private Bitmap limitBitmapSize(Bitmap bmp, int maxDim) {
        if (bmp == null) return null;
        int w = bmp.getWidth(), h = bmp.getHeight();
        if (w <= maxDim && h <= maxDim) return bmp;
        float scale = Math.min((float) maxDim / w, (float) maxDim / h);
        int nw = Math.round(w * scale), nh = Math.round(h * scale);
        Bitmap scaled = Bitmap.createScaledBitmap(bmp, nw, nh, true);
        if (scaled != bmp) bmp.recycle();
        return scaled;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (engine != null) engine.release();
        if (sourceBitmap != null && !sourceBitmap.isRecycled()) sourceBitmap.recycle();
        if (targetBitmap != null && !targetBitmap.isRecycled()) targetBitmap.recycle();
        if (resultBitmap != null && !resultBitmap.isRecycled()) resultBitmap.recycle();
        if (videoTargetBitmap != null && !videoTargetBitmap.isRecycled()) videoTargetBitmap.recycle();
        for (FaceSourceEntry e : faceSourceEntries) {
            if (e.faceImage != null && !e.faceImage.isRecycled()) e.faceImage.recycle();
        }
        for (FaceSourceEntry e : videoFaceSourceEntries) {
            if (e.faceImage != null && !e.faceImage.isRecycled()) e.faceImage.recycle();
        }
        executor.shutdown();
    }
}
        