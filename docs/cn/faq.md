# FAQ

![](../assets/banner.jpg)

## 常见问题

### 如何使用 GPU 加速推理？

MagicMirror 默认使用 CPU 进行推理，暂不支持启用硬件加速选项。

不过你可以尝试 MagicMirror 的 CLI 版本 [TinyFace](https://github.com/idootop/TinyFace)，或者使用 [FaceFusion](https://github.com/facefusion/facefusion)。

### MagicMirror 的体积为什么那么小，是怎么做到的？

这多亏了 [Tauri](https://tauri.app/)，它使用系统自带的 WebView 组件，所以不用像 [Electron](https://www.electronjs.org/) 那样直接把整个 Chromium 浏览器和 Node.js 运行时打包进应用安装包。如果你对此感兴趣，可以在这里了解更多[技术细节](https://tauri.app/start/)。

### MagicMirror 的 LOGO 使用 AI 生成的吗？

是的。MagicMirror 内的所有设计资源：从 Logo、字体到 UI 界面，都是由 AI 生成的 ✨

![](../assets/aigc.webp)

这里我使用的是 [Tensor.ART](https://tusiart.com/)：一个免费的在线 AI 生图网站。像最新的 Flux 和 SD 3.5 等模型都可以直接使用，而且你也可以在线训练自己的模型。

比如 MagicMirror 的 LOGO， 就是我从 Dribbble 上搜集了一些参考图，然后用 Flux 做底模训练的一个模型生成的，非常方便。

![](../assets/train.webp)

相比 [Civitai](https://civitai.com/) 和 [LibLib.AI](https://www.liblib.art/) 等 AI 生图平台，[Tensor.ART](https://tusiart.com/) 的模型数量更多，价格更便宜，性价比应该是这三者里最高的。

如果你也想要尝试 AI 生图，或者正在寻找更有性价比的生图平台，不妨试试看 [Tensor.ART](https://tusiart.com/)。

## 安装问题

### 【macOS】打开 APP 时提示: “MagicMirror”已损坏，无法打开。 你应该将它移到废纸篓。

这是因为 macOS 默认只允许从 App Store 或已知的开发者来源安装应用，以保护用户的安全和隐私。

但是目前我还没有注册苹果开发者账号（$99 一年），所以无法为 MagicMirror 提供有效的签名并上架应用商店。

你可以：

1. 打开访达，然后在右侧菜单栏选择应用程序
2. 找到提示无法验证的应用，右键点击打开即可

![](../assets/macos-open.png)

如果在运行过程中仍然弹窗提示无法打开“xxx”（尤其是 macOS 15 系统版本及以上），

请自行百度 **macOS 如何开启任何来源**，参考：https://www.ghxi.com/jc202309058.html

### 【Windows】提示不是有效的 Win32 应用

MagicMirror 目前只提供了 x64 和 ARM64 架构的安装包，对于较老的 x32 电脑暂不支持。

推荐的运行环境是 Windows 11，如果低于 Windows 10 可能会无法正常运行。

## 运行问题

### 【macOS】卡在启动界面超过 10 分钟

如果你的电脑非 Intel 芯片（比如：M1、M4 芯片），首次启动 APP 时需要进行 Rosatta 转译，大概 3-5 分钟左右。

后续启动将恢复正常时长（5-10s）。如果超过 10 分钟仍未正常启动，请关闭应用后重试。

如果重启后还是无法正常启动，请检查你的 macOS 系统是否不低于 13 版本 (macOS Ventura)，较老的 macOS 系统可能无法正常运行。

查看此处了解更多信息：https://github.com/521xueweihan/HelloGitHub/issues/2859#issuecomment-2562637177

### 【Windows】卡在启动界面超过 10 分钟

如果你在 Windows 系统上一直卡在启动界面，请检查你的 `$HOME/MagicMirror/server.exe` 文件是否存在。

这是使用 [Nuitka](https://github.com/Nuitka/Nuitka) 编译的一个 Python 应用程序，MagicMirror 的正常运行离不开此程序。

![](../assets/windows-home.png)

由于许多病毒软件也喜欢使用 Nuitka 来编译他们的应用程序，从而混淆源代码隐藏自己的恶意行为特征。

所以同样使用 Nuitka 编译的 `server.exe` 文件，在启动时也容易被 Windows 安全应用误标记为病毒文件并删除。

![](../assets/windows-defender.jpg)

你可以在 Windows 安全中心 - 保护历史记录里，手动还原被删除的 `server.exe` 文件。然后重启 MagicMirror 应该就能正常运行了。

> 如果你对 Nuitka 被 Widnows 安全应用误报感兴趣，可以[查看此处](https://github.com/Nuitka/Nuitka/issues/2685#issuecomment-1923357489)了解更多技术细节。

如果仍然启动失败，查看此处了解更多：https://github.com/idootop/MagicMirror/issues/6#issuecomment-2560949972

### 提示换脸失败

你可以换几张其他的图片试试看。比较冷门的图片格式、文件名包含特殊字符、图片分辨率过大等都可能会导致换脸失败。

## 其他问题

### MagicMirror 与 FaceFusion 有什么联系？

简单来说，你可以把 MagicMirror 视为 FaceFusion 的精简版。

我在 [FaceFusion](https://github.com/facefusion/facefusion) 的基础上，移除了所有不必要的模块，只保留了最核心的换脸功能，并由此创造出了 [TinyFace](https://github.com/idootop/TinyFace)：一个超轻量的 Python 换脸工具。

MagicMirror 是在 [TinyFace](https://github.com/idootop/TinyFace) 之上构建的一个 GUI 项目，方便使用。

### MagicMirror 与 InsightFace 有什么联系？

一个经典的换脸工作流，通常由人脸检测、识别、换脸、画质增强等多个步骤组成，不同的步骤依赖不同的模型，而 [InsightFace](https://github.com/deepinsight/insightface/tree/master/examples/in_swapper) 的 `inswapper_128.onnx` 则是换脸过程中使用到的模型，是整个应用的核心。
