"""开发时入口：在项目根目录执行 `python main.py` 启动 uvicorn。

生产环境推荐使用：`uvicorn iskra_engine.api.app:app --host 0.0.0.0 --port 8000`
"""

from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run(
        "iskra_engine.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
