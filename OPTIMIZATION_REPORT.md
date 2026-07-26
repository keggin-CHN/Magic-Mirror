# Magic-Mirror 优化审查报告

> 生成日期：2026-07-26
> 范围：src-python/（Python 后端）、android-app/（Android 端）。前端 src/ 与 src-tauri/ 部分待补充。

---

## 第一部分：Python 后端（src-python/）

### 严重（正确性 / 安全）

#### 1. `PyAVReader.set()` 是空操作，所有依赖 seek 的逻辑全部失效 【功能性 Bug】
- **文件**: `src-python/magic/face.py:72-73`，受影响调用点：`face.py:876, 1978, 2203, 2223, 3036, 3313`
- **问题**: `PyAVReader.set()` 直接 `return False`，不做任何 seek。但代码多处调用 `cap.set(cv2.CAP_PROP_POS_FRAMES, n)` 并假定生效：
  - `detect_face_boxes_in_video`（face.py:1978）：用户选择的关键帧毫秒数被忽略，**永远在第 0 帧检测人脸**；
  - `_swap_face_video_by_sources`（face.py:2203, 2223）：先读"关键帧"（实际是第 0 帧），随后 `set(0)` 无效，读取线程从第 1 帧开始，输出视频丢失第 0 帧且与音轨错位一帧；
  - `_process_deep_video_segment`（face.py:3313）：`cap.set(POS_FRAMES, read_start)` 无效，**每个分段都从第 0 帧读取**，深度换脸的分段并行输出的是重复的视频开头，拼接结果完全错误。
- **修法**: 在 `PyAVReader` 中实现真实 seek（container.seek 后重建 decode 迭代器并丢弃至目标帧）。

#### 2. 桌面版服务默认绑定 `0.0.0.0` 且完全无鉴权
- **文件**: `src-python/server.py:122`、`src-python/magic/app.py:37-45`
- **问题**: 局域网内任何人可提交任务；`inputImage`/`targetFace` 接受任意本地文件路径，可驱动本机读取任意图片/视频。
- **修法**: 默认改为 `127.0.0.1`（Tauri 只需本机通信）。

#### 3. 硬编码 HMAC 默认密钥 + legacy token 中的路径被直接信任
- **文件**: `src-python/web_server.py:145-148, 1746, 1886-1887`；`src-python/magic/app.py:86-88`；`scripts/run-task-config-cli.py:35-37`
- **问题**: `VIDEO_TASK_CONFIG_SECRET` 默认值硬编码；持默认密钥可伪造 legacy `cfg1` token，token 中的 `targetFace.path` 未校验即作为文件路径使用 → 任意文件泄露。
- **修法**: 对 legacy token 提取的路径强制受管目录校验；无环境变量时生成随机密钥并告警。

#### 4. Web 版自动创建默认密码 `123456`
- **文件**: `src-python/web_server.py:169-170`
- **修法**: 删除默认密码回退，要求 `--init-config` + `WEB_INITIAL_PASSWORD` 初始化。

#### 5. 鉴权 token 接受 URL 查询参数传递
- **文件**: `src-python/web_server.py:281`
- **问题**: token 进入访问日志、代理日志、浏览器历史。
- **修法**: 查询参数 token 仅保留给 WebSocket 握手，HTTP 只认 Header/Cookie。

### 高（性能）

#### 6. 每次创建视频任务都对输入视频做完整 SHA256（最多 2GB），且可能算两次
- **文件**: `src-python/magic/app.py:337-340, 404-408`；`src-python/web_server.py:1147-1149, 1219-1223`
- **修法**: 仅在需要 config token 时才计算；verify 与 build 复用同一次哈希。

#### 7. CPU 路径开最多 8 个 worker 线程，但全部串行争抢同一个 `_tf` 实例锁
- **文件**: `src-python/magic/face.py:467, 769`
- **问题**: CPU 模式所有 worker 共享单实例+单锁，推理完全串行，多线程徒增开销。
- **修法**: CPU 模式 worker 数降为 2。

#### 8. GPU 实例池永不释放，最多常驻 4 套完整模型显存
- **文件**: `src-python/magic/face.py:377-450`
- **修法**: 任务完成后按空闲淘汰，或提供显式释放。

#### 9. `PyAVWriter` 把帧率取整为整数，非整数 fps 视频音画渐进失同步
- **文件**: `src-python/magic/face.py:85-87`
- **问题**: 29.97→30、23.976→24，长视频结尾音画错位可达数秒。
- **修法**: 用 `Fraction(fps).limit_denominator()` 作为精确帧率。

#### 10. worker 在持有 `stats_lock` 期间调用 progress_callback
- **文件**: `src-python/magic/face.py:997-1004, 2274-2281`
- **修法**: 锁内拷贝计数，锁外调用回调。

#### 11. `_enhanced_face_config` 在共享实例上有配置竞态
- **文件**: `src-python/magic/face.py:1443-1468`（配合 1050-1058）
- **问题**: 改配置与执行 swap 分两次持锁，CPU 多 worker 下参数会互相污染。
- **修法**: "改配置 + swap + 恢复"放进同一次锁持有。

### 中（代码质量）

#### 12. `app.py` 与 `web_server.py` 约 500 行逻辑复制（进度管理、config 存取、错误简化等），已出现漂移
#### 13. face.py 内部三处重复的区域换脸逻辑 + 两处重复的队列管线辅助函数
#### 14. GPU probe 逻辑三份拷贝（face.py / check_gpu_support.py / scripts/verify-onnxruntime.py）
#### 15. `load_models` 捕获 `BaseException` 并静默返回 False（face.py:177-178）→ 改 `except Exception` + 记日志
#### 16. `PyAVReader.read()` 吞掉所有解码异常，坏帧导致视频静默截断（face.py:58-59）
#### 17. 死代码：`swap_face_video` 未用的 `key_frame_ms` 参数；未引用的 `VIDEO_TASK_CONFIG_TOKEN_PREFIX`；函数内重复 import；`locals().get('save_path')` hack
#### 18. `_get_video_task_config` 读取时刷新 `createdAt`，TTL 形同虚设（web_server.py:1043-1045、app.py:283-284）
#### 19. `_register_result` 失败会让任务永远停在 running（web_server.py:2035）→ 失败时标记任务 failed
#### 20. `_list_library_items` 的 `os.scandir` 迭代器未关闭（web_server.py:859）

### 低（可选）

#### 21. 帧队列内存上限不按分辨率伸缩（face.py:766-777），4K 视频峰值 ~600MB
#### 22. WebSocket 进度推送固定 0.5s 轮询内部状态，可改事件驱动

---

## 第二部分：Android 端（android-app/）

### 严重（错误结果 / 崩溃 / 数据损坏）

#### A1. 模型下载不检查 HTTP 状态码、非原子写入 → 损坏模型被永久当作有效
- **文件**: `MainActivity.java:692-716`（配合 `ModelUtils.java:57-60`）
- **修法**: 检查 `getResponseCode()==200`；先写 `.tmp` 再 `renameTo`。

#### A2. `sourceBitmap` 后台线程使用与主线程 recycle 竞态 → 崩溃
- **文件**: `MainActivity.java:778-784` 与 `805-831`
- **修法**: 检测期间 `setProcessing(true)` 禁用选图，或后台任务持局部引用延迟 recycle。

#### A3. 编码器输入缓冲区不可用时静默丢帧
- **文件**: `VideoProcessor.java:211-219`
- **修法**: `inIdx < 0` 时 drain encoder 后重试，成功 queue 后再推进计数。

#### A4. 完全忽略视频旋转元数据 → 竖屏视频输出横躺、检出率暴跌
- **文件**: `VideoProcessor.java:107-113, 184, 403-415`
- **修法**: 读 `KEY_ROTATION`，`muxer.setOrientationHint(rotation)`。

#### A5. 解码器未请求 YUV420Flexible 颜色格式（部分厂商解码器输出私有 tiled 格式导致 0 帧输出）
- **文件**: `VideoProcessor.java:111-112`
- **修法**: configure 前 `setInteger(KEY_COLOR_FORMAT, COLOR_FormatYUV420Flexible)`。

#### A6. 区域/多人换脸循环中间结果 Bitmap 泄漏
- **文件**: `FaceSwapEngine.java:208-213, 239-243, 282-284, 603, 624`
- **修法**: 覆盖引用前 `if (swapped != result) result.recycle()`。

#### A7. `detectFaces` 缩放检测图有条件泄漏 + 坐标未还原（宽 1921–1939px 边界情况）
- **文件**: `FaceSwapEngine.java:86-94, 827-833`
- **修法**: 回收与坐标还原条件都改为 `det != image`。

#### A8. ONNX Tensor / Result 异常路径不关闭 → native 内存泄漏
- **文件**: `FaceEmbedder.java:59-70`、`FaceEnhancer.java:72-82`
- **修法**: 改 try-with-resources（对照 `FaceDetector.java:103-104` 的正确写法）。

### 性能

#### A9. 嵌套 `float[][][][]` 创建 OnnxTensor，JNI 逐元素拷贝极慢
- **文件**: `ModelUtils.java:572-676`、`FaceSwapper.java:974-975`、`FaceEmbedder.java:59`、`FaceEnhancer.java:72`
- **修法**: 扁平 `float[]` + `FloatBuffer` + `long[] shape`；输出用 `getFloatBuffer()`。

#### A10. `pasteBack`/`blendWithMask` 对全图逐像素混合
- **文件**: `FaceSwapper.java:712-729, 765-803`、`FaceEnhancer.java:201-239`
- **修法**: 用 `inverseMatrix.mapRect()` 求包围盒，仅在子矩形上混合。

#### A11. 视频逐帧大数组分配（YUV↔Bitmap 转换无复用，1080p 每帧 ~20MB 临时分配）
- **文件**: `VideoProcessor.java:466-501, 530`
- **修法**: 单线程 reader/writer 内复用缓冲字段。

#### A12. 编码 PTS 用固定帧率重建，忽略原始时间戳（VFR / fps 元数据缺失时音画不同步）
- **文件**: `VideoProcessor.java:216, 410-411`
- **修法**: `FrameItem` 携带解码 `presentationTimeUs`，编码原样传递。

#### A13. 下载进度每 8KB post 一次主线程 → 节流（`MainActivity.java:704-713`）
#### A14. emap 内存解析阈值 192MB 易 OOM（`FaceSwapper.java:59, 156-163`）→ 统一走流式扫描
#### A15. `swapAllFaces` 每换一张脸就全图重检测（`FaceSwapEngine.java:306-318, 616-627`）

### 内存泄漏与线程管理

#### A16. Activity 销毁后后台任务仍持引用 + `engine.release()` 与 `session.run()` 竞态
- **文件**: `MainActivity.java:1794-1816, 109`
#### A17. `FaceDetector` ThreadLocal 缓冲（每线程 ~6.5MB）在共享线程池场景需 remove（`FaceDetector.java:46-47`）

### 代码质量与安全

#### A18. release 构建静默回退 debug 签名（`app/build.gradle:22-45`）
#### A19. `android:allowBackup="true"`，敏感人脸缓存可被 ADB backup 提取（`AndroidManifest.xml:12`）
#### A20. `saveResult` 输出流异常路径不关闭、MediaStore 脏记录不清理（`MainActivity.java:1624-1628`）
#### A21. 视频输出堆积在 cacheDir 无清理（`FaceSwapEngine.java:509-512`）
#### A22. `MODEL_BASE_URL` 指向疑似不存在的仓库（`MainActivity.java:590`）
#### A23. `KEY_FRAME_RATE` 用 `setFloat`，部分设备 configure 失败（`VideoProcessor.java:187`）→ `setInteger`

### Gradle 构建

#### A24. release 未开启混淆与资源收缩（`minifyEnabled false`）
#### A25. 全局禁用并行构建（`gradle.properties:3-4`），构建速度损失数倍
#### A26. 三个无效的 JdkImageTransform 属性（`gradle.properties:10-12`）→ 删除
#### A27. onnxruntime-android 1.17.0 偏旧；NNAPI EP 在 Android 15+ 已弃用
#### A28. `noCompress` 双写（androidResources 与 aaptOptions 重复）

---

## 第三部分：前端（src/）

### 已修复

#### F1. `LaunchDesktop.tsx:24-40` 内存泄漏：100ms 轮询 interval 在服务器启动失败时永不清除，组件卸载后继续空转 → 改为单 interval + cleanup
#### F2. `Mirror.tsx:833-856` 死代码：整段注释掉的 `remapRegionsForZoom`（24 行）→ 删除
#### F3. `Mirror.tsx:1302` 错误依赖：`addFaceSourcesFromPaths` 的 useCallback 依赖了未使用的 `isStartingSwap`，导致下游回调链（含 useDragDrop onDrop）随 swap 状态频繁重建 → 移除
#### F4. `services/i18n.ts:13` 生产构建也开 `debug: true` → 改 `import.meta.env.DEV`
#### F5. `index.html` 模板残留标题 "Tauri + React + Typescript"、指向不存在的 `/vite.svg` → 修正
#### F6. `services/utils.ts:1-3` 死代码 `timestamp()` → 删除
#### F7. `components/LanguageSwitcher.tsx` 整个组件无引用 → 删除
#### F8. `hooks/useOS.ts` 整个 hook 无引用 → 删除

### 发现但跳过（重构风险大）

- `Mirror.tsx:61-87`：模块级可变对象 `kMirrorStates` + flag 翻转强制重渲染，状态管理反模式 → 建议迁移 xsta/useReducer
- `Mirror.tsx`（~2900 行单组件）：任何 state 变化整树重渲染，建议拆分 + memo
- `hooks/useServer.ts:78-80`：hook 卸载时 `kill()` 服务端进程，建议移到 App 级 window close 事件（需人工确认桌面端导航到 /mirror 后换脸是否正常）
- `services/server.ts` vs `webServer.ts`：约 300 行重复实现
- `package.json`：`i18next-localstorage-backend` 无引用（改 lockfile 会破坏 CI frozen-lockfile，未动）
- `Mirror.tsx:2455`：region 列表用 index 作 key，删除中间项时复用错位
- 多处 `as any` 类型绕过

## 第四部分：Tauri / CI / Docker

### 已修复

#### T1. `commands.rs` `repair_server_runtime` 同步命令在主线程做大量阻塞 I/O（复制 DLL、winget install）导致 UI 冻结 → 改 `#[tauri::command(async)]`
#### T2. `utils.rs` 解压数百 MB server 包阻塞 async 运行时 → `spawn_blocking`
#### T3. `commands.rs` 解压失败时临时 zip 泄漏、清理失败误报错误 → 清理改 best-effort
#### T4. `utils.rs` 下载不检查 HTTP 状态码，404 错误页被当 zip 写盘 → 非 2xx 返回错误
#### T5. `utils.rs` 下载中断半截临时文件泄漏 → 失败路径清理
#### T6. `utils.rs` 每个网络 chunk / zip entry 都 emit 进度事件，IPC 泛洪 → 按整数百分比节流
#### T7. `commands.rs` GUI 程序调用 `where`/`winget` 闪黑色控制台窗口 → `CREATE_NO_WINDOW`
#### T8. `tauriBridge.ts` 僵尸进程：点 Quit 直接 `exit(0)`，Python sidecar 成孤儿进程占用 8023 端口 → 退出前先 `Server.kill()`
#### T9. `Cargo.toml` release 构建带 `devtools` feature（生产可开 F12）→ 移除；新增 `[profile.release]` 体积优化（lto/strip/opt-level=s/panic=abort）
#### T10. `capabilities/default.json` 冗余的裸 `shell:allow-spawn` 声明 → 删除，保留 scoped 版本
#### T11. `build-app.yaml` rust-cache 无 `cache-on-failure`，构建失败缓存全丢 → 加上
#### T12. `docker-build.yml` 裸 `docker build` 无层缓存 → buildx + GHA 层缓存
#### T13. `Dockerfile` 模型下载失败被静默吞掉（循环退出码只看最后一个）→ 每个 wget 失败立即 exit 1
#### T14. `.dockerignore` 补充排除 `__pycache__/`、`*.pyc`

### 发现但跳过

- `tauri.conf.json` `assetProtocol.scope: "**"`：应用需加载用户任选路径的媒体文件，收紧会破坏核心功能
- `capabilities` 允许任意参数 `chmod`：macOS 首次运行需要，收紧有平台兼容风险
- `build-app.yaml` 无 concurrency 组：连续 push 会并行跑多个昂贵构建（行为变化需确认）

### 备注

- CI 模型下载地址 `github.com/idootop/TinyFace/releases/download/models-1.0.0` 为项目原有配置（上游 TinyFace 项目的公开模型），非本次修改引入；如需迁移到自己仓库需先创建 release 上传 4 个模型文件。

---

## 修复优先级总结

| 优先级 | 条目 | 类型 |
|---|---|---|
| P0 | Python #1 seek 失效（深度换脸分段结果错误） | 正确性 |
| P0 | Python #2/#3/#4 无鉴权监听、默认密钥、默认密码 | 安全 |
| P0 | Android A1-A6（下载损坏 / 崩溃竞态 / 丢帧 / 旋转 / 颜色格式 / 泄漏） | 正确性 |
| P1 | Python #6/#7/#9、Android A9/A10/A11 | 性能 |
| P2 | 其余代码质量与构建配置项 | 质量 |
