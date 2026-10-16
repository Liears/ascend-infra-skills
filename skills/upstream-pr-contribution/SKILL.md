---
name: upstream-pr-contribution
description: 把一个本地已验证的昇腾优化变成上游（vllm-ascend 等）可合入的 PR——分支基点纪律、最小 diff、DCO、CI 修复循环、上游测试契约、合入标签门。用于"上游定价实验证明价值之后"的贡献回传阶段。
---

# upstream-pr-contribution

上游 PR 不是"把补丁推上去"。本 skill 编码从**本地已验证的优化**到**上游 CI 全绿、等维护者合入**之间的全部流程纪律，每一条都来自一次真实的 8 轮 CI 修复战役（vllm-ascend PR #18967）。

## 前置条件（不满足就不要开工）

- 优化已在本地硬件验证过（性能 + 精度都有数字）
- 已跑过上游定价实验（见 `extreme-perf-campaign` 的 P5）：确认上游没有等价实现、gap 真实存在
- 想清楚开关形态：**opt-in 环境变量 / 配置项，默认关 = 上游行为逐位不变**。无法做到"默认关零影响"的改动先拆分

## 流程

### 1. 分支与基点

- **基点 = 当日的上游 main**（`git fetch --depth 50 <upstream> main`），不是最新 release tag。
  从 tag 切分支会产生幽灵 diff：GitHub 把"main 领先 tag 的部分"算成你的回退——实际案例中 3 文件 +549 的改动显示成 213 文件 +14144。
- 上游 https 直连不稳时重试 fetch，或对三个文件级 diff 用 API 逐文件比对 blob sha。
- 单 commit（squash）比多 commit 更利于评审；每个 commit 必须 DCO 签署。

### 2. 最小化 diff

以"当日 main 的文件 + 最小增量"重建，不要拿本地工作树直接覆盖：

- 剥离全部调试脚手架：探针、dump、A/B 臂、热切换文件、判定实验开关
- 多个内部 env 旋钮合并为**一个上游风格开关**
- 注释全部英文、写根因（评审价值在"为什么"，不在"改了什么"）
- 重新做三方合并（base=你验证时的上游快照, ours=你的补丁, other=当日 main），冲突逐个解
- 每步改动后 `python -m py_compile` + `pyflakes`（undefined 检查）+ **AST 函数清单对照 main**——盲切代码可能把函数定义切丢而编译仍然通过

### 3. 上游测试契约（最容易漏的一步）

**先查 `tests/ut/` 下有没有你触碰模块的既有用例**——它们就是行为契约。真实案例：dspark proposer 有 30+ 个单测，编码了旧 executor 合同、桶尺寸、初始化形状；改行为不同步改测试 = cpu-ut 全红。

- 改行为的 PR 同步更新其行为测试是正当且必需的
- 测试 mock 常用 `__new__` 绕过 `__init__` 构造实例——新增实例属性时用 `getattr(self, ..., default)` 容错，别假设 `__init__` 跑过

### 4. 提交与 PR 文案

- 标题：`[Type][Component] summary`（读 `git log --oneline -30` 学惯例）
- 正文按上游 PR 模板三段：What/Why、user-facing change、How tested（硬件数字 + 可复现命令 + 口径）
- checklist 诚实作答（没写回归测试就说明为什么）
- **`git commit --amend -m` 会丢 Signed-off-by**——amend 后必须 `--amend -s --no-edit` 再推

### 5. CI 修复循环

上游 pre-commit 通常含 ruff check/format、mypy、codespell、typos、gitleaks 等十余个 hook：

- 本地装同版本工具预检（lint 项本地能复现的先修掉）
- **同版本号 ≠ 同行为**：真实案例中本地 ruff 0.14.0 format 通过、CI 同版本号仍要改——此时别再猜，从 CI 日志把它的补丁**逐字抄回来**应用
- mypy 的 override 检查会波及你未触碰的子类（改了基类签名 → 所有覆写子类都要补参数）
- E402：新增的模块级 env 定义不要挡在 import 块前面

### 6. 合入标签门

外部贡献者的终点是"所有自动检查绿 + ci-gate 因缺标签而红"。触碰源码的 PR 需要维护者加 `ready-precise` / `ready-all` / `main2main` 标签才会跑合入前选定测试——annotation 会写明。到这一步就是等，不要反复 force-push。

## 凭证（共享机器）

- fine-grained PAT（`github_pat_` 前缀）**不能**在他人组织的仓库创建 PR——平台限制，换 classic PAT（`repo` scope 即可）
- classic PAT 走个人 `GH_CONFIG_DIR`，GraphQL 常因缺 `read:org` 失败，**用 REST**（`gh api repos/<org>/<repo>/pulls -f ...`）

## 工具

- `scripts/check_touch.py` —— 输入改动文件列表，列出 `tests/ut/` 下的相关既有测试（提交前自查测试契约）并输出 PR 前检查清单

## 红线

- 没跑定价实验就提 PR（上游可能已有等价实现）
- PR 声称的性能数字不是 PR 代码跑出来的（清理后的代码必须重新上板验证）
- force-push 解决标签门（那是维护者的门，不是 CI 故障）
