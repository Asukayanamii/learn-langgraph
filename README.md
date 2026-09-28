# LangGraph 从入门到能写 Agent

一套**能跑、能懂、能改**的 LangGraph 中文学习项目。

十节课 + 一个完整实战项目，从"第一个聊天机器人"一路走到"自己设计多智能体系统"。
**不需要 API Key 也能跑通全部内容**，学完你能独立写出自己的 Agent 项目。

---

## 三十秒上手

挑一种你习惯的方式，任选其一即可。

### 方式一：uv（最快，推荐）

[uv](https://docs.astral.sh/uv/) 是目前最快的 Python 包管理器，全过程约 10 秒。

```bash
uv venv                             # 创建 .venv（约 0.2 秒）
uv pip install -r requirements.txt  # 装依赖（约 10 秒）
uv run verify_all.py                # 一键跑通全部课程
```

还没装 uv？

```powershell
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 方式二：pip + venv（Python 自带，无需额外工具）

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
python verify_all.py
```

### 方式三：conda

```bash
conda create -n learn-langgraph python=3.12 -y
conda activate learn-langgraph
pip install -r requirements.txt
python verify_all.py
```

---

看到 `全部通过！11/11 个脚本正常运行` 就说明环境没问题。

```bash
# 想自己跟它对话？
python lessons/01_basic_chatbot.py --interactive
# 用 uv 的话：
uv run lessons/01_basic_chatbot.py --interactive
```

> **不需要 API Key。** 项目内置了一个「离线模拟模型」，
> 它和真模型的接口完全一样，支持工具调用、流式输出、结构化输出。
> 所以你能把全部机制都真跑一遍，而不是只看代码。
> 想换成真模型？把 `.env.example` 复制成 `.env` 填上 Key 即可，**代码一行不用改**。

---

## 为什么这个项目不一样

| 常见教程 | 这个项目 |
|---|---|
| 只给代码，不给原理 | 每节课的注释是**讲解稿**；`docs/` 有 9 篇原理文档 |
| 必须配好 API Key 才能跑 | **零配置可跑**，先学明白再决定要不要花钱 |
| 只会用 `create_agent` 一行流 | 第 10 课**把封装全拆掉**，手写一遍 |
| 学完不知道能干什么 | 有**完整实战项目**，并能照着改成自己的 |
| 遇到报错就卡住 | 有**真实踩坑记录的排错文档** |

---

## 课程地图

```
    第一阶段：跑起来                    第二阶段：做扎实
    ┌────────────────────────┐         ┌────────────────────────┐
    │ 01 基础聊天机器人       │         │ 04 人机协同（审批）      │
    │ 02 工具调用（ReAct）    │  ───▶   │ 05 自定义状态与 Reducer  │
    │ 03 记忆持久化           │         │ 06 流式输出             │
    └────────────────────────┘         └────────────────────────┘
              │                                   │
              └───────────────┬───────────────────┘
                              ▼
    第三阶段：上强度                    第四阶段：看透本质
    ┌────────────────────────┐         ┌────────────────────────┐
    │ 07 控制流（分支/并行）  │         │ 10 手写 Agent（拆封装） │
    │ 08 子图与时间旅行       │  ───▶   │ 实战 命令行智能助理      │
    │ 09 多智能体（Supervisor）│         │                        │
    └────────────────────────┘         └────────────────────────┘
```

### 每课学什么

| # | 课程 | 你会掌握 | 核心概念 |
|---|---|---|---|
| 01 | [基础聊天机器人](lessons/01_basic_chatbot.py) | 把程序写成一张图 | `StateGraph`、`State`、节点、边、`compile()` |
| 02 | [工具调用](lessons/02_tools.py) | 让模型能用工具 | `@tool`、`bind_tools`、`ToolNode`、**ReAct 循环** |
| 03 | [记忆持久化](lessons/03_memory.py) | 让它记住你说的话 | `checkpointer`、`thread_id`、状态快照 |
| 04 | [人机协同](lessons/04_human_in_the_loop.py) | 危险操作先问你 | `interrupt()`、`Command(resume=...)` |
| 05 | [自定义状态](lessons/05_custom_state.py) | 设计数据契约 | 多字段 State、**自定义 reducer** |
| 06 | [流式输出](lessons/06_streaming.py) | 打字机效果 | 四种 `stream_mode`、`get_stream_writer` |
| 07 | [控制流](lessons/07_control_flow.py) | 分支、并行、动态分发 | 条件边、**超级步**、`Send` map-reduce |
| 08 | [子图与时间旅行](lessons/08_subgraph.py) | 复用与回滚 | 子图、`get_state_history`、`update_state` |
| 09 | [多智能体](lessons/09_multi_agent.py) | 分工协作 | **Supervisor**、结构化输出、`recursion_limit` |
| 10 | [手写 Agent](lessons/10_handmade_agent.py) | 看透本质 | 亲手实现 `ToolNode` / `tools_condition` |
| ★ | [实战项目](capstone/assistant.py) | 做一个真东西 | 上述能力的完整组合 |

---

## 项目结构

```
learn-langgraph/
│
├── README.md                    ← 你在这里
├── requirements.txt             依赖清单
├── .env.example                 配置模板（想接真模型就复制成 .env）
├── verify_all.py                一键验证全部课程
│
├── lessons/                     ★ 十节课程，每节都能独立运行
│   ├── 01_basic_chatbot.py          StateGraph 三要素 + reducer
│   ├── 02_tools.py                  ReAct 循环
│   ├── 03_memory.py                 Checkpointer 记忆
│   ├── 04_human_in_the_loop.py      人工审批
│   ├── 05_custom_state.py           状态设计与自定义 reducer
│   ├── 06_streaming.py              四种流模式
│   ├── 07_control_flow.py           分支 / 并行 / Send
│   ├── 08_subgraph.py               子图 / 时间旅行
│   ├── 09_multi_agent.py            Supervisor 多智能体
│   └── 10_handmade_agent.py         手写 Agent 循环
│
├── docs/                        ★ 九篇原理文档（讲"为什么"）
│   ├── 00-学习路线.md               从哪里开始、怎么学
│   ├── 01-执行模型与状态.md          超级步、通道、reducer 的底层原理
│   ├── 02-消息协议与工具调用.md      消息协议、工具调用全流程
│   ├── 03-持久化与中断原理.md        Checkpointer 与 interrupt 的机制
│   ├── 04-流式输出原理.md            流式数据从哪来
│   ├── 05-控制流与并行原理.md        调度规则与三条铁律
│   ├── 06-多智能体架构.md            四种拓扑与选型决策
│   ├── 07-手写Agent-ReAct原理.md     ★ 从零推导一个 Agent
│   └── 08-常见问题与排错.md          ★ 真实踩坑记录
│
├── capstone/                    ★ 综合实战：命令行智能助理
│   ├── README.md                    设计说明与改造指南
│   ├── assistant.py                 主程序（图组装 + 会话管理 + CLI）
│   └── tools.py                     工具集
│
└── common/                      公共脚手架（看了就懂，不用背）
    ├── config.py                    读取 .env
    ├── fake_model.py                离线模拟模型（零配置可跑的关键）
    ├── llm.py                       统一的"取模型"入口
    ├── pretty.py                    终端美化输出
    └── graph_viz.py                 把图打印成结构图
```

---

## 怎么学

### 推荐节奏

```
① 跑       python lessons/0X_xxx.py
              ↓
② 读       读代码注释（写得很详细，就是讲解稿）
              ↓
③ 懂       读 docs/ 里对应的原理文档
              ↓
④ 练       ★ 做每节课最后的「动手练习」—— 这一步最关键
              ↓
⑤ 玩       --interactive 进交互模式，亲眼看它的表现
```

**⚠️ 只做①②不做④，等于没学。**

### 两种运行模式

```bash
python lessons/01_basic_chatbot.py                # 自动演示（看效果）
python lessons/01_basic_chatbot.py --interactive  # 交互模式（自己说话）
```

---

## 配置真实大模型（可选）

**不配也能学完**，但配了能体验到真正的智能。

```bash
copy .env.example .env      # Windows
cp .env.example .env        # macOS / Linux
```

编辑 `.env`：

```ini
LLM_PROVIDER=custom
LLM_API_KEY=sk-你的key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

**支持任何 OpenAI 兼容接口**：

| 服务商 | `LLM_BASE_URL` |
|---|---|
| DeepSeek | `https://api.deepseek.com/v1` |
| Moonshot | `https://api.moonshot.cn/v1` |
| 阿里通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` |
| 本地 Ollama | `http://localhost:11434/v1` |

> ⚠️ 模型需要支持 **function calling**（工具调用）。
> 老旧的补全模型不行。

---

## 学完之后你能做什么

- ✅ 读懂任何基于 LangGraph 的 Agent 项目
- ✅ 独立设计有状态、多步骤、可持久化的 AI 应用
- ✅ 给你的 Agent 加上工具、记忆、人工审批、流式输出
- ✅ 设计多智能体协作系统
- ✅ 知道 `ToolNode`、`tools_condition`、`create_agent` 内部在做什么
- ✅ 遇到报错知道去哪查（而不是只能重装依赖）

**自测清单**在 [`docs/00-学习路线.md`](docs/00-学习路线.md) 的最后一节——能答出来才算真学会。

---

## 环境要求

| 项 | 要求 |
|---|---|
| Python | 3.10 及以上（本项目的验证环境是 3.14） |
| 操作系统 | Windows / macOS / Linux 都可以 |
| 网络 | 只在首次 `pip install` 时需要 |
| API Key | **不需要**（除非你要接真模型） |

### 依赖版本

```
langgraph>=1.0,<2.0                图引擎
langgraph-checkpoint-sqlite>=3.0   SQLite 持久化
langchain>=1.0,<2.0                模型抽象层
langchain-openai>=1.0              OpenAI 兼容接口
python-dotenv>=1.0                 读取 .env
```

本项目在 **LangGraph 1.2.11 / LangChain 1.4.0** 上完整验证通过。

---

## 常见问题

**Q：真的不用 API Key 吗？**
真的。`common/fake_model.py` 实现了一个支持工具调用、流式输出、结构化输出的模拟模型。
它靠规则匹配，没有真模型的智能，但**所有 LangGraph 的机制都能跑通**——
因为你学的是"怎么编排"，不是"模型多聪明"。

**Q：跑出来一堆 `InvalidUpdateError`？**
去看 [`docs/08-常见问题与排错.md`](docs/08-常见问题与排错.md) 第 2.1 节，
这是并行分支没配 reducer 的经典问题。

**Q：`python lessons/xx.py` 报 `No module named 'common'`？**
看排错文档 1.1 节。简单说：要保证 `lessons/_bootstrap.py` 被 import 到。

**Q：能直接拿来当自己项目的基础吗？**
可以。`capstone/` 就是按"能改成真实项目"的思路写的，
换工具、换存储、换人设都有说明（见 `capstone/README.md`）。

**Q：为什么用离线模型时，回答看起来傻傻的？**
因为它只是规则匹配器，不是真的语言模型。它的作用是**让你不花钱就能验证代码逻辑**。
接上真模型后，同样的代码会立刻变聪明。

---

## 许可

本项目为教学用途编写，代码可自由取用、修改、用于你自己的项目。

课程内容参考 [LangGraph 官方文档](https://langgraph.com.cn) 的知识体系，
但所有代码、示例、讲解均为本项目原创。

---

**准备好了吗？**

```bash
python lessons/01_basic_chatbot.py
```
