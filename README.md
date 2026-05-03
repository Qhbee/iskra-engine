# iskra-engine

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Status](https://img.shields.io/badge/Status-Active-success)

---

`iskra-engine` 是 `iskra` 的智能引擎 🧠，采用 [iskra-data](https://github.com/Qhbee/iskra-data) 的数据 。

---

## Start

在项目根目录安装可编辑模式后启动：

```bash
uv pip install -e .
python main.py
```

或直接使用 uvicorn：

```bash
uvicorn iskra_engine.api.app:app --reload --host 127.0.0.1 --port 8000
```

健康检查：`GET http://127.0.0.1:8000/health`
API 文档（Swagger UI）：`GET http://127.0.0.1:8000/docs`
API 文档（ReDoc）：`GET http://127.0.0.1:8000/redoc`
