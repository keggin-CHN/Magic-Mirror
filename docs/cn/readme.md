# MagicMirror

一键 AI 换脸，发现更美的自己 ✨

[![GitHub release](https://img.shields.io/github/v/release/idootop/MagicMirror.svg)](https://github.com/idootop/MagicMirror/releases) [![Build APP](https://github.com/idootop/MagicMirror/actions/workflows/build-app.yaml/badge.svg)](https://github.com/idootop/MagicMirror/actions/workflows/build-app.yaml) [![Build Server](https://github.com/idootop/MagicMirror/actions/workflows/build-server.yaml/badge.svg)](https://github.com/idootop/MagicMirror/actions/workflows/build-server.yaml)

![](../assets/banner.jpg)

## 功能亮点

- 一键换脸：打开安装包，拖张照片进去就能换脸，无需配置各种复杂参数。
- 超低门槛：不用 GPU 也能运行，普通的小白电脑也可以轻松玩转 AI 换脸。
- 隐私安全：完全在你的电脑本地运行，不需要联网，不用担心你的图片会被上传到任何地方。
- 极致精简：安装包不到 10 MB，模型文件加起来不到 1 GB，这可能是最轻量的离线换脸应用之一。

## 快速开始

> 🔥 演示视频：https://www.bilibili.com/video/BV1TTzfYDEUe

MagicMirror 仅支持 macOS 13（Ventura）和 Windows 10 及以上版本系统：

1. [安装教程](./install.md)
2. [使用教程](./usage.md)
3. [常见问题](./faq.md)（90%的问题可以在这里找到答案）

如果你有其他问题，请提交 [Issue](https://github.com/idootop/MagicMirror/issues)。

## 动机

![391785246-b3b52898-4d43-40db-8fbe-acbc00d78eec](https://github.com/user-attachments/assets/6500a393-69e7-42c9-bf78-febc84d7e5e5)


我想你跟我一样，有时会纠结：自己适合哪种发型，或者哪种穿搭最好看？

要是有一个应用，可以把我们看到的喜欢的发型或心动的穿搭，直接放到自己的身上预览效果，那真是太好了。

现在的 AI 技术已经很成熟了，但是市面上大部分的 AI 换脸应用：

- 要么需要配置复杂的参数和运行环境，要有性能足够强劲的 GPU 才能运行，使用门槛和成本偏高；
- 要么需要上传自己的图片到服务器进行转换，存在隐私泄漏的风险。

理想的解决方案应该像自拍一样简单：不需要设置复杂的参数，不用购买昂贵的设备，也无需担心隐私泄漏的问题。

所以，为什么不自己做一个呢？

于是便有了 MagicMirror ✨

Enjoy! ;)

## 鸣谢

MagicMirror 的实现主要使用和参考了以下开源项目，特此鸣谢:

- [TinyFace](https://github.com/idootop/TinyFace): The minimalist face swapping tool that just works. 
- [FaceFusion](https://github.com/facefusion/facefusion): Industry leading face manipulation platform
- [InsightFace](https://github.com/deepinsight/insightface): State-of-the-art 2D and 3D Face Analysis Project
- [Nuitka](https://github.com/Nuitka/Nuitka): Nuitka is a Python compiler written in Python.
- [Tauri](https://github.com/tauri-apps/tauri): Build smaller, faster, and more secure desktop and mobile applications with a web frontend.

## 免责声明

MagicMirror 仅限个人娱乐与创意用途，严禁用于商业用途。请注意：

- **道德使用**：本软件不得用于以下行为，包括但不限于：a) 恶意冒充他人，b) 散布虚假信息，c) 侵犯个人隐私或尊严，d) 制作淫秽或不当内容。
- **内容版权**：用户应对以下内容负责：a) 获取使用源图像的必要许可，b) 尊重版权及知识产权，c) 遵守当地关于 AI 生成内容的法律法规。
- **免责声明**：用户应对生成的内容以及由其使用引发的任何法律、道德或个人问题承担全部责任。本软件及其开发者对用户生成的内容不承担任何责任。

使用 MagicMirror 即表示您已阅读并同意上述条款，并承诺负责任地使用本软件。

## License

MIT License © 2025-PRESENT [Del Wang](https://del.wang)
