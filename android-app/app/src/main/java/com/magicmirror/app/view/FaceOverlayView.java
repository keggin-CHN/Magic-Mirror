package com.magicmirror.app.view;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

/**
 * 人脸检测结果叠加视图 + 区域选择功能。
 * 叠加在 ImageView 上方，绘制检测到的人脸框和用户手动选择的区域。
 */
public class FaceOverlayView extends View {

    public interface OnRegionSelectedListener {
        void onRegionSelected(RectF region);
    }

    public interface OnFaceClickedListener {
        void onFaceClicked(int faceIndex, RectF faceBox);
    }

    // 检测到的人脸框（图片坐标系）
    private final List<RectF> faceBoxes = new ArrayList<>();
    // 用户选择的区域（图片坐标系）
    private final List<RectF> selectedRegions = new ArrayList<>();
    // 高亮的人脸索引
    private final List<Integer> highlightedFaces = new ArrayList<>();

    // 源图尺寸（用于坐标映射）
    private int imageWidth, imageHeight;
    // 显示区域（View 内图片实际绘制区域）
    private RectF displayRect = new RectF();

    // 绘制工具
    private final Paint faceBoxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint faceBoxHighlightPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint regionPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint regionFillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelBgPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    // 区域选择状态
    private boolean regionSelectMode = false;
    private float dragStartX, dragStartY, dragEndX, dragEndY;
    private boolean isDragging = false;

    private OnRegionSelectedListener regionListener;
    private OnFaceClickedListener faceClickListener;

    public FaceOverlayView(Context context) { super(context); init(); }
    public FaceOverlayView(Context context, AttributeSet attrs) { super(context, attrs); init(); }
    public FaceOverlayView(Context context, AttributeSet attrs, int defStyle) { super(context, attrs, defStyle); init(); }

    private void init() {
        faceBoxPaint.setStyle(Paint.Style.STROKE);
        faceBoxPaint.setColor(0xFF00E676);
        faceBoxPaint.setStrokeWidth(3f);

        faceBoxHighlightPaint.setStyle(Paint.Style.STROKE);
        faceBoxHighlightPaint.setColor(0xFFFFD600);
        faceBoxHighlightPaint.setStrokeWidth(4f);

        regionPaint.setStyle(Paint.Style.STROKE);
        regionPaint.setColor(0xFFE94560);
        regionPaint.setStrokeWidth(3f);
        regionPaint.setPathEffect(new android.graphics.DashPathEffect(new float[]{12, 6}, 0));

        regionFillPaint.setStyle(Paint.Style.FILL);
        regionFillPaint.setColor(0x22E94560);

        labelPaint.setColor(Color.WHITE);
        labelPaint.setTextSize(28f);
        labelPaint.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);

        labelBgPaint.setStyle(Paint.Style.FILL);
        labelBgPaint.setColor(0xCC000000);
    }

    // ==================== 公开 API ====================

    public void setImageSize(int w, int h) {
        imageWidth = w;
        imageHeight = h;
        updateDisplayRect();
        invalidate();
    }

    public void setFaceBoxes(List<RectF> boxes) {
        faceBoxes.clear();
        if (boxes != null) faceBoxes.addAll(boxes);
        invalidate();
    }

    public void highlightFace(int index) {
        if (!highlightedFaces.contains(index)) {
            highlightedFaces.add(index);
            invalidate();
        }
    }

    public void clearHighlights() {
        highlightedFaces.clear();
        invalidate();
    }

    public void addSelectedRegion(RectF region) {
        selectedRegions.add(region);
        invalidate();
    }

    public void clearSelectedRegions() {
        selectedRegions.clear();
        invalidate();
    }

    public void clearAll() {
        faceBoxes.clear();
        selectedRegions.clear();
        highlightedFaces.clear();
        invalidate();
    }

    public void setRegionSelectMode(boolean enabled) {
        regionSelectMode = enabled;
    }

    public boolean isRegionSelectMode() {
        return regionSelectMode;
    }

    public void setOnRegionSelectedListener(OnRegionSelectedListener l) { regionListener = l; }
    public void setOnFaceClickedListener(OnFaceClickedListener l) { faceClickListener = l; }

    public List<RectF> getFaceBoxes() { return new ArrayList<>(faceBoxes); }
    public List<RectF> getSelectedRegions() { return new ArrayList<>(selectedRegions); }

    // ==================== 绘制 ====================

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        updateDisplayRect();
    }

    private void updateDisplayRect() {
        if (imageWidth <= 0 || imageHeight <= 0) return;
        int vw = getWidth(), vh = getHeight();
        if (vw <= 0 || vh <= 0) return;

        // centerCrop 模式的映射
        float scaleX = (float) vw / imageWidth;
        float scaleY = (float) vh / imageHeight;
        float scale = Math.max(scaleX, scaleY); // centerCrop 用 max
        float dw = imageWidth * scale, dh = imageHeight * scale;
        float dx = (vw - dw) / 2f, dy = (vh - dh) / 2f;
        displayRect.set(dx, dy, dx + dw, dy + dh);
    }

    private RectF imageToView(RectF imgRect) {
        if (imageWidth <= 0 || imageHeight <= 0) return imgRect;
        float scale = Math.max((float) getWidth() / imageWidth, (float) getHeight() / imageHeight);
        float dx = (getWidth() - imageWidth * scale) / 2f;
        float dy = (getHeight() - imageHeight * scale) / 2f;
        return new RectF(
            imgRect.left * scale + dx, imgRect.top * scale + dy,
            imgRect.right * scale + dx, imgRect.bottom * scale + dy
        );
    }

    private RectF viewToImage(RectF viewRect) {
        if (imageWidth <= 0 || imageHeight <= 0) return viewRect;
        float scale = Math.max((float) getWidth() / imageWidth, (float) getHeight() / imageHeight);
        float dx = (getWidth() - imageWidth * scale) / 2f;
        float dy = (getHeight() - imageHeight * scale) / 2f;
        return new RectF(
            (viewRect.left - dx) / scale, (viewRect.top - dy) / scale,
            (viewRect.right - dx) / scale, (viewRect.bottom - dy) / scale
        );
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        // 绘制检测到的人脸框
        for (int i = 0; i < faceBoxes.size(); i++) {
            RectF vr = imageToView(faceBoxes.get(i));
            boolean hl = highlightedFaces.contains(i);
            canvas.drawRect(vr, hl ? faceBoxHighlightPaint : faceBoxPaint);

            // 标签
            String label = "#" + (i + 1);
            float tw = labelPaint.measureText(label);
            float lx = vr.left, ly = vr.top - 6;
            canvas.drawRect(lx, ly - 30, lx + tw + 12, ly + 4, labelBgPaint);
            canvas.drawText(label, lx + 6, ly, labelPaint);
        }

        // 绘制已选区域
        for (int i = 0; i < selectedRegions.size(); i++) {
            RectF vr = imageToView(selectedRegions.get(i));
            canvas.drawRect(vr, regionFillPaint);
            canvas.drawRect(vr, regionPaint);

            String label = "R" + (i + 1);
            float tw = labelPaint.measureText(label);
            canvas.drawRect(vr.left, vr.bottom - 34, vr.left + tw + 12, vr.bottom, labelBgPaint);
            canvas.drawText(label, vr.left + 6, vr.bottom - 6, labelPaint);
        }

        // 绘制正在拖拽的区域
        if (isDragging) {
            float l = Math.min(dragStartX, dragEndX), t = Math.min(dragStartY, dragEndY);
            float r = Math.max(dragStartX, dragEndX), b = Math.max(dragStartY, dragEndY);
            RectF dr = new RectF(l, t, r, b);
            canvas.drawRect(dr, regionFillPaint);
            canvas.drawRect(dr, regionPaint);
        }
    }

    // ==================== 触摸事件 ====================

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (regionSelectMode) {
            return handleRegionDrag(event);
        } else {
            return handleFaceClick(event);
        }
    }

    private boolean handleRegionDrag(MotionEvent event) {
        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                dragStartX = event.getX();
                dragStartY = event.getY();
                dragEndX = dragStartX;
                dragEndY = dragStartY;
                isDragging = true;
                invalidate();
                return true;
            case MotionEvent.ACTION_MOVE:
                dragEndX = event.getX();
                dragEndY = event.getY();
                invalidate();
                return true;
            case MotionEvent.ACTION_UP:
                isDragging = false;
                float l = Math.min(dragStartX, dragEndX), t = Math.min(dragStartY, dragEndY);
                float r = Math.max(dragStartX, dragEndX), b = Math.max(dragStartY, dragEndY);
                if (r - l > 20 && b - t > 20) {
                    RectF viewRect = new RectF(l, t, r, b);
                    RectF imgRect = viewToImage(viewRect);
                    // 裁剪到图片范围
                    imgRect.left = Math.max(0, imgRect.left);
                    imgRect.top = Math.max(0, imgRect.top);
                    imgRect.right = Math.min(imageWidth, imgRect.right);
                    imgRect.bottom = Math.min(imageHeight, imgRect.bottom);
                    selectedRegions.add(imgRect);
                    if (regionListener != null) regionListener.onRegionSelected(imgRect);
                }
                invalidate();
                return true;
        }
        return false;
    }

    private boolean handleFaceClick(MotionEvent event) {
        if (event.getAction() != MotionEvent.ACTION_UP) return true;
        float x = event.getX(), y = event.getY();
        for (int i = 0; i < faceBoxes.size(); i++) {
            RectF vr = imageToView(faceBoxes.get(i));
            if (vr.contains(x, y)) {
                if (faceClickListener != null) faceClickListener.onFaceClicked(i, faceBoxes.get(i));
                return true;
            }
        }
        return true;
    }
}