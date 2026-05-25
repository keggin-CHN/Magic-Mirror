# Contributing to Magic Mirror

感谢你对 Magic Mirror 的关注！

## 开发环境

1. 克隆仓库
2. 安装 Node.js 18+ 和 pnpm
3. 安装 Python 3.10+ 和依赖：`pip install -r src-python/requirements.txt`
4. 启动前端：`pnpm dev`
5. 启动后端：`make dev`

## 代码规范

- Python 代码使用 `ruff check` 和 `ruff format` 格式化
- 所有函数必须添加 docstring
- 保持代码简洁、可读

## 提交 PR

1. 确保 CI 检查全部通过（Lint、Build Web、Build Server）
2. 更新文档（如有需要）
3. 每个 PR 只做一件事
4. 描述清楚 PR 的目的

## 报告问题

使用 GitHub Issue 模板报告 Bug 或请求新功能。
