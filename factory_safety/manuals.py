"""검증된 안전 매뉴얼 기반 RAG (근거 없는 질문에는 답하지 않고 거부한다).

    data/manuals/*.md  →  load_chunks()  →  build_index()  →  retrieve(query)

매뉴얼 문서(data/manuals/)는 공개된 산업안전 자료(KOSHA 등)를 참고해 사람이 큐레이션한 것이다.
LLM이 안전 문장을 직접 지어내지 않도록, 답변은 항상 이 문서에서 검색된 내용에 근거해야 한다
(근거 문서가 없거나 유사도가 낮으면 retrieve()가 빈 리스트를 돌려주고, 호출 측이 "거부 + 관리자 호출"로 분기한다).

load_chunks()는 numpy/표준 라이브러리만 써서 Isaac 쪽 파이썬(.venv)에서도 테스트할 수 있다.
build_index()/retrieve()는 langchain-ollama + langchain-chroma가 필요해서 .venv-assistant 안에서만 쓴다
(llm_agent.py 와 같은 이유로 지연 import).
"""
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANUALS_DIR = os.path.join(ROOT, "data", "manuals")
INDEX_DIR = os.path.join(ROOT, "assets", "generated", "manuals_index")
EMBED_MODEL = "nomic-embed-text"
RELEVANCE_MIN = 0.55       # 이보다 낮은 유사도면 "근거 없음" (거부 → 관리자 호출)
TOP_K = 4

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_value(v):
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        return [x.strip() for x in v[1:-1].split(",") if x.strip()]
    return v


def _parse_frontmatter(text):
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        meta[k.strip()] = _parse_value(v)
    return meta, m.group(2)


def _split_sections(body):
    """'## 제목' 기준으로 나눈다: [(제목, 본문)]. 맨 앞 제목 없는 부분은 버린다."""
    parts = re.split(r"\n##\s+", "\n" + body.strip())
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        head, _, rest = p.partition("\n")
        out.append((head.strip(), rest.strip()))
    return out


def load_chunks(manuals_dir=MANUALS_DIR):
    """data/manuals/*.md -> [{id, title, section, text, source, category, applies_to}].
    매뉴얼 문서를 ## 단위로 쪼개서, 검색 결과가 짧고 구체적인 조항이 되게 한다."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(manuals_dir, "*.md"))):
        with open(path, encoding="utf-8") as f:
            meta, body = _parse_frontmatter(f.read())
        doc_id = meta.get("id", os.path.splitext(os.path.basename(path))[0])
        for section, text in _split_sections(body):
            if not text:
                continue
            chunks.append({
                "id": f"{doc_id}#{section}",
                "doc_id": doc_id,
                "title": meta.get("title", doc_id),
                "section": section,
                "text": text,
                "source": meta.get("source", ""),
                "category": meta.get("category", ""),
                "applies_to": meta.get("applies_to", []),
            })
    return chunks


def build_index(manuals_dir=MANUALS_DIR, index_dir=INDEX_DIR, embed_model=EMBED_MODEL, rebuild=False):
    """로컬 Chroma 벡터스토어를 만들거나(없으면) 열어서(있으면) 돌려준다."""
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_ollama import OllamaEmbeddings

    emb = OllamaEmbeddings(model=embed_model)
    if rebuild and os.path.isdir(index_dir):
        import shutil
        shutil.rmtree(index_dir)
    fresh = rebuild or not os.path.isdir(index_dir) or not os.listdir(index_dir)
    store = Chroma(collection_name="manuals", embedding_function=emb, persist_directory=index_dir)
    if fresh:
        chunks = load_chunks(manuals_dir)
        # Chroma 메타데이터는 str/int/float/bool 만 허용 (list 불가) -> applies_to 는 쉼표 문자열로
        docs = [Document(page_content=c["text"],
                         metadata={k: (",".join(v) if isinstance(v, list) else v) for k, v in c.items() if k != "text"})
                for c in chunks]
        if docs:
            store.add_documents(docs, ids=[c["id"] for c in chunks])
    return store


def retrieve(query, k=TOP_K, threshold=RELEVANCE_MIN, index_dir=INDEX_DIR, embed_model=EMBED_MODEL):
    """질의 -> [{text, title, section, source, score}] (유사도 threshold 미달이면 빈 리스트 = 근거 없음).
    근거가 없으면 호출 측(LLM 노드)은 답변을 거부하고 관리자 호출로 넘겨야 한다."""
    store = build_index(index_dir=index_dir, embed_model=embed_model)
    try:
        hits = store.similarity_search_with_relevance_score(query, k=k)
    except Exception:
        hits = [(d, 1.0) for d in store.similarity_search(query, k=k)]
    out = []
    for doc, score in hits:
        if score < threshold:
            continue
        out.append({"text": doc.page_content, "title": doc.metadata.get("title"), "section": doc.metadata.get("section"),
                    "source": doc.metadata.get("source"), "score": round(float(score), 3)})
    return out


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "방치된 유출 조치 방법"
    for r in retrieve(q):
        print(f"[{r['score']}] {r['title']} / {r['section']}\n  {r['text'][:120]}...")
