package com.magicmirror.app.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.ViewParent;

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
    private final RectF displayRect = new RectF();

    // 绘制工具
    private final Paint faceBoxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint faceBoxHighlightPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint regionPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint regionSelectedPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint regionFillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelBgPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    // 区域选择状态
    private boolean regionSelectMode = false;
    private boolean showFaceBoxes = true;
    private float dragStartX, dragStartY, dragEndX, dragEndY;
    private boolean isDragging = false;
    private boolean isMovingRegion = false;
    private float lastTouchX, lastTouchY;
    private int selectedRegionIndex = -1;
    private boolean pendingRegionMove = false;
    private boolean isResizingRegion = false;
    private int resizeHandle = HANDLE_NONE;
    private long downEventTime = 0L;

    private static final int HANDLE_NONE = 0;
    private static final int HANDLE_MOVE = 1;
    private static final int HANDLE_LEFT = 2;
    private static final int HANDLE_TOP = 3;
    private static final int HANDLE_RIGHT = 4;
    private static final int HANDLE_BOTTOM = 5;
    private static final int HANDLE_TOP_LEFT = 6;
    private static final int HANDLE_TOP_RIGHT = 7;
    private static final int HANDLE_BOTTOM_LEFT = 8;
    private static final int HANDLE_BOTTOM_RIGHT = 9;

    private OnRegionSelectedListener regionListener;
    private OnFaceClickedListener faceClickListener;

    public FaceOverlayView(Context context) {
        super(context);
        init();
    }

    public FaceOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public FaceOverlayView(Context context, AttributeSet attrs, int defStyle) {
        super(context, attrs, defStyle);
        init();
    }

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
        regionPaint.setPathEffect(new android.graphics.DashPathEffect(new float[] { 12, 6 }, 0));

        regionSelectedPaint.setStyle(Paint.Style.STROKE);
        regionSelectedPaint.setColor(0xFFFFD600);
        regionSelectedPaint.setStrokeWidth(4f);
        regionSelectedPaint.setPathEffect(new android.graphics.DashPathEffect(new float[] { 12, 6 }, 0));

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
        if (boxes != null) {
            faceBoxes.addAll(boxes);
        }
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
        selectedRegionIndex = selectedRegions.size() - 1;
        invalidate();
    }

    public RectF removeSelectedRegion() {
        if (selectedRegionIndex < 0 || selectedRegionIndex >= selectedRegions.size()) {
            return null;
        }
        RectF removed = selectedRegions.remove(selectedRegionIndex);
        if (selectedRegions.isEmpty()) {
            selectedRegionIndex = -1;
        } else {
            selectedRegionIndex = Math.min(selectedRegionIndex, selectedRegions.size() - 1);
        }
        invalidate();
        return removed;
    }

    public void clearSelectedRegions() {
        selectedRegions.clear();
        selectedRegionIndex = -1;
        isDragging = false;
        isMovingRegion = false;
        pendingRegionMove = false;
        isResizingRegion = false;
        resizeHandle = HANDLE_NONE;
        invalidate();
    }

    public void clearAll() {
        faceBoxes.clear();
        selectedRegions.clear();
        highlightedFaces.clear();
        selectedRegionIndex = -1;
        isDragging = false;
        isMovingRegion = false;
        pendingRegionMove = false;
        isResizingRegion = false;
        resizeHandle = HANDLE_NONE;
        invalidate();
    }

    public void setRegionSelectMode(boolean enabled) {
        regionSelectMode = enabled;
        isDragging = false;
        isMovingRegion = false;
        pendingRegionMove = false;
        isResizingRegion = false;
        resizeHandle = HANDLE_NONE;
    }

    public boolean isRegionSelectMode() {
        return regionSelectMode;
    }

    public void setShowFaceBoxes(boolean show) {
        if (showFaceBoxes != show) {
            showFaceBoxes = show;
            invalidate();
        }
    }

    public boolean isShowFaceBoxes() {
        return showFaceBoxes;
    }

    public void setOnRegionSelectedListener(OnRegionSelectedListener l) {
        regionListener = l;
    }

    public void setOnFaceClickedListener(OnFaceClickedListener l) {
        faceClickListener = l;
    }

    public List<RectF> getFaceBoxes() {
        return new ArrayList<>(faceBoxes);
    }

    public List<RectF> getSelectedRegions() {
        return new ArrayList<>(selectedRegions);
    }

    // ==================== 绘制 ====================

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        updateDisplayRect();
    }

    private void updateDisplayRect() {
        if (imageWidth <= 0 || imageHeight <= 0) {
            return;
        }
        int vw = getWidth(), vh = getHeight();
        if (vw <= 0 || vh <= 0) {
            return;
        }

        // centerCrop 模式的映射
        float scaleX = (float) vw / imageWidth;
        float scaleY = (float) vh / imageHeight;
        float scale = Math.max(scaleX, scaleY); // centerCrop 用 max
        float dw = imageWidth * scale, dh = imageHeight * scale;
        float dx = (vw - dw) / 2f, dy = (vh - dh) / 2f;
        displayRect.set(dx, dy, dx + dw, dy + dh);
    }

    private RectF imageToView(RectF imgRect) {
        if (imageWidth <= 0 || imageHeight <= 0) {
            return imgRect;
        }
        float scale = Math.max((float) getWidth() / imageWidth, (float) getHeight() / imageHeight);
        float dx = (getWidth() - imageWidth * scale) / 2f;
        float dy = (getHeight() - imageHeight * scale) / 2f;
        return new RectF(
                imgRect.left * scale + dx, imgRect.top * scale + dy,
                imgRect.right * scale + dx, imgRect.bottom * scale + dy);
    }

    private RectF viewToImage(RectF viewRect) {
        if (imageWidth <= 0 || imageHeight <= 0) {
            return viewRect;
        }
        float scale = Math.max((float) getWidth() / imageWidth, (float) getHeight() / imageHeight);
        float dx = (getWidth() - imageWidth * scale) / 2f;
        float dy = (getHeight() - imageHeight * scale) / 2f;
        return new RectF(
                (viewRect.left - dx) / scale, (viewRect.top - dy) / scale,
                (viewRect.right - dx) / scale, (viewRect.bottom - dy) / scale);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        // 绘制检测到的人脸框
        if (showFaceBoxes) {
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
        }

        // 绘制已选区域
        for (int i = 0; i < selectedRegions.size(); i++) {
            RectF vr = imageToView(selectedRegions.get(i));
            canvas.drawRect(vr, regionFillPaint);
            canvas.drawRect(vr, i == selectedRegionIndex ? regionSelectedPaint : regionPaint);

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
                ViewParent parent = getParent();
                if (parent != null) {
                    parent.requestDisallowInterceptTouchEvent(true);
                }

                int hit = findRegionAtViewPoint(event.getX(), event.getY());
                if (hit >= 0) {
                    selectedRegionIndex = hit;
                    isDragging = false;
                    isMovingRegion = false;
                    pendingRegionMove = false;
                    isResizingRegion = false;
                    resizeHandle = findResizeHandleAtViewPoint(hit, event.getX(), event.getY());
                    if (resizeHandle != HANDLE_NONE) {
                        isResizingRegion = true;
                    } else {
                        pendingRegionMove = true;
                        resizeHandle = HANDLE_MOVE;
                    }
                    downEventTime = event.getEventTime();
                    lastTouchX = event.getX();
                    lastTouchY = event.getY();
                    invalidate();
                    return true;
                }

                isMovingRegion = false;
                pendingRegionMove = false;
                isResizingRegion = false;
                resizeHandle = HANDLE_NONE;
                dragStartX = event.getX();
                dragStartY = event.getY();
                dragEndX = dragStartX;
                dragEndY = dragStartY;
                isDragging = true;
                invalidate();
                return true;

            case MotionEvent.ACTION_MOVE:
                ViewParent moveParent = getParent();
                if (moveParent != null) {
                    moveParent.requestDisallowInterceptTouchEvent(true);
                }

                if (isResizingRegion && selectedRegionIndex >= 0) {
                    resizeRegionByViewPoint(selectedRegionIndex, resizeHandle, event.getX(), event.getY());
                    lastTouchX = event.getX();
                    lastTouchY = event.getY();
                    invalidate();
                } else if ((pendingRegionMove || isMovingRegion) && selectedRegionIndex >= 0) {
                    long pressElapsed = event.getEventTime() - downEventTime;
                    if (!isMovingRegion && pendingRegionMove
                            && pressElapsed >= ViewConfiguration.getLongPressTimeout()) {
                        isMovingRegion = true;
                        pendingRegionMove = false;
                    }
                    if (isMovingRegion) {
                        float x = event.getX();
                        float y = event.getY();
                        moveRegionByViewDelta(selectedRegionIndex, x - lastTouchX, y - lastTouchY);
                        lastTouchX = x;
                        lastTouchY = y;
                        invalidate();
                    }
                } else if (isDragging) {
                    dragEndX = event.getX();
                    dragEndY = event.getY();
                    invalidate();
                }
                return true;

            case MotionEvent.ACTION_UP:
                ViewParent upParent = getParent();
                if (upParent != null) {
                    upParent.requestDisallowInterceptTouchEvent(false);
                }

                if (isResizingRegion) {
                    isResizingRegion = false;
                    resizeHandle = HANDLE_NONE;
                    return true;
                }

                if (isMovingRegion || pendingRegionMove) {
                    isMovingRegion = false;
                    pendingRegionMove = false;
                    resizeHandle = HANDLE_NONE;
                    return true;
                }

                if (!isDragging) {
                    return true;
                }

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
                    selectedRegionIndex = selectedRegions.size() - 1;
                    if (regionListener != null) {
                        regionListener.onRegionSelected(imgRect);
                    }
                }
                invalidate();
                return true;

            case MotionEvent.ACTION_CANCEL:
                ViewParent cancelParent = getParent();
                if (cancelParent != null) {
                    cancelParent.requestDisallowInterceptTouchEvent(false);
                }
                isDragging = false;
                isMovingRegion = false;
                pendingRegionMove = false;
                isResizingRegion = false;
                resizeHandle = HANDLE_NONE;
                return true;
        }
        return false;
    }

    private boolean handleFaceClick(MotionEvent event) {
        int action = event.getAction();
        if (action == MotionEvent.ACTION_DOWN) {
            ViewParent parent = getParent();
            if (parent != null) {
                parent.requestDisallowInterceptTouchEvent(true);
            }
            return true;
        }
        if (action == MotionEvent.ACTION_CANCEL) {
            ViewParent parent = getParent();
            if (parent != null) {
                parent.requestDisallowInterceptTouchEvent(false);
            }
            return true;
        }
        if (action != MotionEvent.ACTION_UP) {
            return true;
        }

        ViewParent parent = getParent();
        if (parent != null) {
            parent.requestDisallowInterceptTouchEvent(false);
        }

        float x = event.getX(), y = event.getY();

        int hitRegion = findRegionAtViewPoint(x, y);
        if (hitRegion >= 0) {
            selectedRegionIndex = hitRegion;
            invalidate();
            return true;
        }

        selectedRegionIndex = -1;
        if (!showFaceBoxes) {
            invalidate();
            return true;
        }

        for (int i = 0; i < faceBoxes.size(); i++) {
            RectF vr = imageToView(faceBoxes.get(i));
            if (vr.contains(x, y)) {
                if (faceClickListener != null) {
                    faceClickListener.onFaceClicked(i, faceBoxes.get(i));
                }
                return true;
            }
        }
        invalidate();
        return true;
    }

    private void moveRegionByViewDelta(int regionIndex, float dxView, float dyView) {
        if (regionIndex < 0 || regionIndex >= selectedRegions.size()) {
            return;
        }
        if (imageWidth <= 0 || imageHeight <= 0) {
            return;
        }

        float scale = Math.max((float) getWidth() / imageWidth, (float) getHeight() / imageHeight);
        if (scale <= 0f) {
            return;
        }

        RectF region = selectedRegions.get(regionIndex);
        float dx = dxView / scale;
        float dy = dyView / scale;
        region.offset(dx, dy);

        float clampDx = 0f;
        if (region.left < 0f) {
            clampDx = -region.left;
        } else if (region.right > imageWidth) {
            clampDx = imageWidth - region.right;
        }

        float clampDy = 0f;
        if (region.top < 0f) {
            clampDy = -region.top;
        } else if (region.bottom > imageHeight) {
            clampDy = imageHeight - region.bottom;
        }

        if (clampDx != 0f || clampDy != 0f) {
            region.offset(clampDx, clampDy);
        }
    }

    private int findRegionAtViewPoint(float x, float y) {
        for (int i = selectedRegions.size() - 1; i >= 0; i--) {
            RectF vr = imageToView(selectedRegions.get(i));
            if (vr.contains(x, y)) {
                return i;
            }
        }
        return -1;
    }

    private int findResizeHandleAtViewPoint(int regionIndex, float x, float y) {
        if (regionIndex < 0 || regionIndex >= selectedRegions.size()) {
            return HANDLE_NONE;
        }

        RectF vr = imageToView(selectedRegions.get(regionIndex));
        float r = dpToPx(18f);

        if (distance(x, y, vr.left, vr.top) <= r) {
            return HANDLE_TOP_LEFT;
        }
        if (distance(x, y, vr.right, vr.top) <= r) {
            return HANDLE_TOP_RIGHT;
        }
        if (distance(x, y, vr.left, vr.bottom) <= r) {
            return HANDLE_BOTTOM_LEFT;
        }
        if (distance(x, y, vr.right, vr.bottom) <= r) {
            return HANDLE_BOTTOM_RIGHT;
        }

        if (Math.abs(x - vr.left) <= r && y >= vr.top - r && y <= vr.bottom + r) {
            return HANDLE_LEFT;
        }
        if (Math.abs(x - vr.right) <= r && y >= vr.top - r && y <= vr.bottom + r) {
            return HANDLE_RIGHT;
        }
        if (Math.abs(y - vr.top) <= r && x >= vr.left - r && x <= vr.right + r) {
            return HANDLE_TOP;
        }
        if (Math.abs(y - vr.bottom) <= r && x >= vr.left - r && x <= vr.right + r) {
            return HANDLE_BOTTOM;
        }
        return HANDLE_NONE;
    }

    private void resizeRegionByViewPoint(int regionIndex, int handle, float viewX, float viewY) {
        if (regionIndex < 0 || regionIndex >= selectedRegions.size()) {
            return;
        }
        if (imageWidth <= 0 || imageHeight <= 0 || getWidth() <= 0 || getHeight() <= 0) {
            return;
        }

        RectF region = selectedRegions.get(regionIndex);
        float ix = viewToImageX(viewX);
        float iy = viewToImageY(viewY);

        float oldLeft = region.left;
        float oldTop = region.top;
        float oldRight = region.right;
        float oldBottom = region.bottom;

        switch (handle) {
            case HANDLE_LEFT:
                region.left = ix;
                break;
            case HANDLE_TOP:
                region.top = iy;
                break;
            case HANDLE_RIGHT:
                region.right = ix;
                break;
            case HANDLE_BOTTOM:
                region.bottom = iy;
                break;
            case HANDLE_TOP_LEFT:
                region.left = ix;
                region.top = iy;
                break;
            case HANDLE_TOP_RIGHT:
                region.right = ix;
                region.top = iy;
                break;
            case HANDLE_BOTTOM_LEFT:
                region.left = ix;
                region.bottom = iy;
                break;
            case HANDLE_BOTTOM_RIGHT:
                region.right = ix;
                region.bottom = iy;
                break;
            default:
                return;
        }

        float scale = Math.max((float) getWidth() / imageWidth, (float) getHeight() / imageHeight);
        float minSize = Math.max(8f, dpToPx(28f) / Math.max(0.01f, scale));

        if (region.left < 0f)
            region.left = 0f;
        if (region.top < 0f)
            region.top = 0f;
        if (region.right > imageWidth)
            region.right = imageWidth;
        if (region.bottom > imageHeight)
            region.bottom = imageHeight;

        if (region.width() < minSize) {
            if (isLeftSideHandle(handle)) {
                region.left = region.right - minSize;
            } else if (isRightSideHandle(handle)) {
                region.right = region.left + minSize;
            } else {
                region.left = oldLeft;
                region.right = oldRight;
            }
        }

        if (region.height() < minSize) {
            if (isTopSideHandle(handle)) {
                region.top = region.bottom - minSize;
            } else if (isBottomSideHandle(handle)) {
                region.bottom = region.top + minSize;
            } else {
                region.top = oldTop;
                region.bottom = oldBottom;
            }
        }

        if (region.left < 0f) {
            region.left = 0f;
            if (region.right < minSize)
                region.right = minSize;
        }
        if (region.top < 0f) {
            region.top = 0f;
            if (region.bottom < minSize)
                region.bottom = minSize;
        }
        if (region.right > imageWidth) {
            region.right = imageWidth;
            if (region.left > imageWidth - minSize)
                region.left = imageWidth - minSize;
        }
        if (region.bottom > imageHeight) {
            region.bottom = imageHeight;
            if (region.top > imageHeight - minSize)
                region.top = imageHeight - minSize;
        }
    }

    private boolean isLeftSideHandle(int handle) {
        return handle == HANDLE_LEFT || handle == HANDLE_TOP_LEFT || handle == HANDLE_BOTTOM_LEFT;
    }

    private boolean isRightSideHandle(int handle) {
        return handle == HANDLE_RIGHT || handle == HANDLE_TOP_RIGHT || handle == HANDLE_BOTTOM_RIGHT;
    }

    private boolean isTopSideHandle(int handle) {
        return handle == HANDLE_TOP || handle == HANDLE_TOP_LEFT || handle == HANDLE_TOP_RIGHT;
    }

    private boolean isBottomSideHandle(int handle) {
        return handle == HANDLE_BOTTOM || handle == HANDLE_BOTTOM_LEFT || handle == HANDLE_BOTTOM_RIGHT;
    }

    private float viewToImageX(float vx) {
        float scale = Math.max((float) getWidth() / imageWidth, (float) getHeight() / imageHeight);
        float dx = (getWidth() - imageWidth * scale) / 2f;
        return (vx - dx) / scale;
    }

    private float viewToImageY(float vy) {
        float scale = Math.max((float) getWidth() / imageWidth, (float) getHeight() / imageHeight);
        float dy = (getHeight() - imageHeight * scale) / 2f;
        return (vy - dy) / scale;
    }

    private float dpToPx(float dp) {
        return dp * getResources().getDisplayMetrics().density;
    }

    private float distance(float x1, float y1, float x2, float y2) {
        float dx = x1 - x2;
        float dy = y1 - y2;
        return (float) Math.sqrt(dx * dx + dy * dy);
    }
}