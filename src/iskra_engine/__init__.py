"""iskra-engine：通过对内的 HTTP API 提供 RAG 服务。"""

from importlib.metadata import version

# 版本唯一写在 pyproject.toml；此处读的是安装后的包元数据（.dist-info）。
# 改 version 后需在本项目根执行一次：uv sync 或 uv pip install -e .
__version__ = version("iskra-engine")
