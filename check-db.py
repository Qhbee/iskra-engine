# check_db_clean.py
# 一次性连通性/ pgvector 烟测。
# 凭据从环境变量读取；本地可建 .env（见 .env.example），勿提交 .env
import os
import psycopg
from dotenv import load_dotenv

# 在仓库根/当前工作目录加载 .env（有则读，无则只读系统环境变量）
load_dotenv()

# psycopg3: 用 dbname，不用 database
def _db_config() -> dict:
    """与 libpq 一致：PGHOST / PGPORT / PGDATABASE / PGUSER / PGPASSWORD。"""
    return {
        "host": os.environ.get("PGHOST", "127.0.0.1"),
        "port": os.environ.get("PGPORT", "5432"),
        "dbname": os.environ.get("PGDATABASE", "core_db"),
        "user": os.environ.get("PGUSER", "db_admin"),
        "password": os.environ.get("PGPASSWORD", "password"),
    }

# 格式：ssh -L 本地端口:127.0.0.1:远程容器端口 用户@服务器IP
# ssh -L 5432:127.0.0.1:5432 root@<你的服务器IP>

def check_and_clean():
    conn = None
    try:
        print("🔌 正在通过隧道连接数据库...")
        conn = psycopg.connect(**_db_config())
        cur = conn.cursor()

        # 1. 检查版本
        cur.execute("SELECT version();")
        print(f"✅ 数据库连接成功: {cur.fetchone()[0].split()[0]}")

        # 2. 检查/安装 pgvector 插件
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        print("✅ Vector (pgvector) 插件状态正常")

        # 3. 确保 iskra 模式存在
        cur.execute("CREATE SCHEMA IF NOT EXISTS iskra;")
        print("✅ Schema 'iskra' 检查/创建完毕")

        # 4. 在 iskra 模式下建表
        table_name = "iskra.connection_test"
        cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (id bigserial PRIMARY KEY, vec vector(3));")

        # 5. 写入测试
        cur.execute(f"INSERT INTO {table_name} (vec) VALUES ('[1,2,3]');")
        print(f"✅ 向量写入测试成功 (表: {table_name})")

        # 6. 清理：删测试表，并删掉为烟测临时建的 schema，避免库上残留空 iskra
        cur.execute(f"DROP TABLE IF EXISTS {table_name};")
        cur.execute("DROP SCHEMA IF EXISTS iskra CASCADE;")
        conn.commit()
        print("✅ 现场清理完毕 (测试表与 schema iskra 已删除)")

    except Exception as e:
        print(f"❌ 连接失败，错误: {e}")
        print("请检查：1. SSH 隧道是否已开启？ 2. 密码是否正确？")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()


if __name__ == "__main__":
    check_and_clean()