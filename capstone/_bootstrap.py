"""和 lessons/_bootstrap.py 同样的作用：把项目根目录加入模块搜索路径，
这样 `python capstone/assistant.py` 才能 import 到 common 包。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
