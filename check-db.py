# check_db_clean.py
import psycopg2

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": "5432",
    "database": "core_db",
    "user": "db_admin",     # 使用时改为真实用户
    "password": "password", # 使用时改为真实密码
    # 关键点：连接参数里虽然不能直接写 schema，但在 execute 时可以指定
}

# 格式：ssh -L 本地端口:127.0.0.1:远程容器端口 用户@服务器IP
# ssh -L 5432:127.0.0.1:5432 root@<你的服务器IP>

def check_and_clean():
    conn = None
    try:
        print("🔌 正在通过隧道连接数据库...")
        conn = psycopg2.connect(**DB_CONFIG)
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

        # 6. 清理现场 (Drop Table)
        cur.execute(f"DROP TABLE {table_name};")
        conn.commit()
        print("✅ 现场清理完毕 (测试表已删除)")

    except Exception as e:
        print(f"❌ 连接失败，错误: {e}")
        print("请检查：1. SSH 隧道是否已开启？ 2. 密码是否正确？")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()


if __name__ == "__main__":
    check_and_clean()