-- iskra-engine: 初始表结构（MVP 两表）

-- extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- 可选：专用于本项目的库内命名空间，避免和别的应用挤在 public
-- CREATE SCHEMA IF NOT EXISTS iskra;

-- 本会话里未加限定名的对象默认落在此 schema
-- SET search_path TO iskra, public;

-- 篇目 = 一个 index.md（或你约定的单文件）
CREATE TABLE IF NOT EXISTS document (
    id             bigserial PRIMARY KEY,
    rel_path       text NOT NULL,
    title          text,
    book           text,
    full_text      text NOT NULL,
    content_sha256 char(64) NOT NULL,
    updated_at     timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT document_rel_path_unique UNIQUE (rel_path)
);

-- 单篇下多块；embedding 为 Jina v5 text-small 默认 1024 维
CREATE TABLE IF NOT EXISTS chunk (
    id           bigserial PRIMARY KEY,
    document_id  bigint NOT NULL REFERENCES document (id) ON DELETE CASCADE,
    chunk_index  integer NOT NULL,
    text         text NOT NULL,
    embedding    vector(1024) NOT NULL,
    CONSTRAINT chunk_document_chunk_unique UNIQUE (document_id, chunk_index)
);

-- 向量检索：余弦（与归一化后的 embedding 向量配合；查询用 <=> 排序）
CREATE INDEX IF NOT EXISTS chunk_embedding_hnsw_idx
    ON chunk
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 下列单列索引与 UNIQUE(document_id, chunk_index) 自动生成的 B-tree 在「最左列 document_id」上功能重叠：
-- 按 document_id 查块、与 JOIN 规划都可用该联合（唯一）索引的最左前缀，再建 (document_id) 多占一份维护与空间，故默认不创建。
-- 外键列：按篇拉块 / 级联删时规划器友好
-- CREATE INDEX IF NOT EXISTS chunk_document_id_idx ON chunk (document_id);

COMMENT ON TABLE document IS '文章篇目，对应一个 markdown 文件；rel_path 在库内唯一';
COMMENT ON COLUMN document.id IS '主键，bigserial 插入时自动生成';
COMMENT ON COLUMN document.rel_path IS '相对语料根目录的路径，唯一标识磁盘上的该篇';
COMMENT ON COLUMN document.title IS '篇名，来自 frontmatter 等，可空';
COMMENT ON COLUMN document.book IS '书/卷级标题，如列宁全集第几卷，可空';
COMMENT ON COLUMN document.full_text IS 'markdown 全文（含 frontmatter），与磁盘一致';
COMMENT ON COLUMN document.content_sha256 IS '去掉 frontmatter 后正文的 SHA-256 十六进制；与上次相同则本段不删 chunk/不重嵌，只正文变更才重切分重嵌。';
COMMENT ON COLUMN document.updated_at IS '更新时间';

COMMENT ON TABLE chunk IS '同篇文章可能切分为多块，每块一条向量供检索';
COMMENT ON COLUMN chunk.id IS '主键，bigserial 插入时自动生成';
COMMENT ON COLUMN chunk.document_id IS '外键，指向 document.id，删篇时级联删块';
COMMENT ON COLUMN chunk.chunk_index IS '该篇内第几块，自 0 递增；与 UNIQUE(document_id, chunk_index) 防重复';
COMMENT ON COLUMN chunk.text IS '本块给向量与 LLM 用的纯文本（可带标题行前缀，由切分逻辑决定）';
COMMENT ON COLUMN chunk.embedding IS 'Jina v5 text-small 向量，维数 1024；勿混用其它模型，换模型需全量重嵌';

COMMENT ON INDEX chunk_embedding_hnsw_idx IS 'HNSW 余弦，检索 ORDER BY embedding <=> 查询向量';
-- 若取消上面 chunk_document_id_idx 的注释并建索引，可在此补
-- COMMENT ON INDEX chunk_document_id_idx IS '按 document_id 查块、与 JOIN 规划';

