# 2026-05-28 Git 归档与 .gitignore 配置

## 完成事项

### 1. 创建 .gitignore

项目之前没有 `.gitignore`，导致 `git add` 时把模型缓存（HuggingFace cache）、嵌入 git 仓库等不该提交的内容都暂存了。

创建的 `.gitignore` 排除项：

| 类别 | 规则 | 原因 |
|------|------|------|
| 模型缓存 | `cache/`、`**/cache/` | HuggingFace 下载的模型文件，数百 MB~数 GB |
| 模型权重 | `*.safetensors`、`*.bin`、`*.pt`、`*.gguf` 等 | 大文件，不适合 git |
| Python | `__pycache__/`、`.venv/` | 编译缓存和虚拟环境 |
| IDE | `.vscode/`、`.idea/` | 编辑器配置 |
| 训练产物 | `output/`、`wandb/` | 训练 checkpoint 和日志 |
| 环境变量 | `.env` | 防止泄露密钥 |
| OpenAI 缓存 | `openai_cache/` | API 响应缓存 |

### 2. 移除嵌入 git 仓库

`06-微调LLM写唐诗/GenAI-Hw5/` 内含 `.git` 目录，被 git 识别为 embedded repo（子模块）。

处理方式：
- 删除 `GenAI-Hw5/.git` 目录
- 将其内容（README + 唐诗训练/测试数据）作为普通文件提交
- 从 `.gitignore` 中移除 `**/GenAI-Hw5/` 排除规则

### 3. 提交记录

两次 commit：

1. `feat: add Phase 0-1 study materials (modules 01-08)` — 24 个文件，7212 行新增
2. `feat: add GenAI-Hw5 reference repo (removed embedded .git)` — 5 个文件，25089 行新增

---

## 踩坑记录

### git add 把 cache 目录暂存

**问题**：`git add "05-部署语言模型/"` 把 `cache/models--*/` 下的模型文件全暂存了（几百 MB）。

**解决**：先创建 `.gitignore`，再 `git reset HEAD` 取消暂存，重新 `git add`。

### git index.lock 被占用

**问题**：后台 `git add` 任务还在运行时，执行 `git reset` 报 `index.lock` 锁文件冲突。

**解决**：用 `TaskStop` 停掉后台任务，再删除 `.git/index.lock`。

### embedded git repo 警告

**问题**：`git add` 时 git 提示 `adding embedded git repository`，建议用 `git submodule add`。

**解决**：删除内嵌 `.git` 目录，作为普通文件提交。对于学习项目中的参考仓库，submodule 过于复杂，普通文件更简单。

---

## 当前进度

| Phase | 状态 |
|-------|------|
| 0 | ✅ Week 1 完成，Week 2 部分 |
| 1 | 🔄 Week 3 进行中（Guide 07/08 已迁移） |
| 2-5 | ⬜ |

**下一步**：完成 Phase 1 Week 3 — 迁移 Guide 10（Top-K/Top-P）+ Beam Search 运行验证
