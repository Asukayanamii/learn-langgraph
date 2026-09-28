"""让课程脚本可以"直接运行"的小引导。

背景知识（很实用，值得一看）：
    Python 运行一个脚本时，只会把这个脚本**所在的那一层目录**加进模块搜索路径。
    所以当你执行 `python lessons/01_basic_chatbot.py` 时，
    搜索路径里是 `lessons/`，而不是项目根目录，
    于是 `from common import pretty` 就会报 ModuleNotFoundError。

解决办法：在导入 common 之前，先手动把项目根目录塞进 sys.path。

每个课程文件开头写一行 `import _bootstrap` 即可（_bootstrap 就在 lessons/ 目录里，
能被直接找到）。导入这个模块时，下面的代码就会自动执行。
"""

import sys
from pathlib import Path

# 本文件在 lessons/ 下，.parent.parent 就是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
