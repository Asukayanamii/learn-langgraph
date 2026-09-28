"""课程目录。

每节课都是一个**可以独立运行**的 Python 文件，从 01 到 10 循序渐进：

    01_basic_chatbot.py     最小可用的聊天机器人（StateGraph 三要素）
    02_tools.py             让机器人会用工具（ReAct 循环）
    03_memory.py            让它记住你说过的话（Checkpointer）
    04_human_in_the_loop.py 让它动手前先问你（interrupt 人工审批）
    05_custom_state.py      设计自己的状态结构（多字段 + 自定义 reducer）
    06_streaming.py         把输出变成"打字机"效果（4 种流模式）
    07_control_flow.py      条件分支、并行执行、动态分发（Send）
    08_subgraph.py          把图当积木复用（子图）
    09_multi_agent.py       多个专家协作（Supervisor 多智能体）
    10_handmade_agent.py    不用任何封装，手写一个 Agent 循环（彻底搞懂原理）

运行方式：
    python lessons/01_basic_chatbot.py                # 自动演示
    python lessons/01_basic_chatbot.py --interactive   # 进入交互模式
"""
