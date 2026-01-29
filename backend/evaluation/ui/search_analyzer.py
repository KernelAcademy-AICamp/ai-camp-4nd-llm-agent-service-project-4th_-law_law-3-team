"""
검색 분석 UI

실시간 벡터 검색 테스트 및 분석:
1. 쿼리 입력 및 검색 실행
2. 검색 결과 시각화 (유사도 점수)
3. 문서 상세 보기
4. 검색 결과를 Ground Truth로 바로 추가
"""

import time
from pathlib import Path
from typing import Optional

import gradio as gr


_embedding_model = None
_vector_store = None


def _get_embedding_model_name() -> str:
    """중앙 설정에서 임베딩 모델명 가져오기"""
    try:
        from app.core.config import settings
        return settings.LOCAL_EMBEDDING_MODEL
    except ImportError:
        import os
        return os.getenv("LOCAL_EMBEDDING_MODEL", "nlpai-lab/KURE-v1")


def get_embedding_model():
    """임베딩 모델 로드 (lazy, .env 설정 사용)"""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = _get_embedding_model_name()
        _embedding_model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
        )
    return _embedding_model


def get_vector_store():
    """벡터 스토어 연결 (lazy)"""
    global _vector_store
    if _vector_store is None:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from app.common.vectorstore.lancedb import LanceDBStore
        _vector_store = LanceDBStore()
    return _vector_store


def search_vectors(
    query: str,
    top_k: int = 10,
    doc_type_filter: str = "all",
) -> tuple[str, str, str]:
    """
    벡터 검색 수행

    Returns:
        (결과 마크다운, 메트릭 정보, 상세 JSON)
    """
    if not query.strip():
        return "쿼리를 입력해주세요.", "", ""

    model = get_embedding_model()
    store = get_vector_store()

    start_time = time.perf_counter()
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    where_filter = None
    if doc_type_filter == "판례":
        where_filter = {"data_type": "판례"}
    elif doc_type_filter == "법령":
        where_filter = {"data_type": "법령"}

    result = store.search(
        query_embedding=query_embedding,
        n_results=top_k,
        where=where_filter,
    )

    latency_ms = (time.perf_counter() - start_time) * 1000

    if not result.ids or not result.ids[0]:
        return "검색 결과가 없습니다.", f"Latency: {latency_ms:.2f}ms", ""

    lines = []
    details = []
    precedent_count = 0
    law_count = 0

    for i, doc_id in enumerate(result.ids[0]):
        meta = result.metadatas[0][i] if result.metadatas else {}
        content = result.documents[0][i] if result.documents else ""
        distance = result.distances[0][i] if result.distances else 0.0
        score = 1.0 - distance

        data_type = meta.get("data_type", "")
        title = meta.get("title", "")
        source_id = meta.get("source_id", doc_id)

        if data_type == "판례":
            precedent_count += 1
        else:
            law_count += 1

        score_class = "score-high" if score > 0.8 else ("score-low" if score < 0.5 else "")

        lines.append(f"### #{i+1} [{source_id}]")
        lines.append(f"**{title}** | <span class='{score_class}'>Score: {score:.4f}</span>")
        lines.append(f"*{data_type}* | {meta.get('source_name', '')} | {meta.get('date', '')}")
        lines.append(f"```\n{content[:300]}...\n```")
        lines.append("")

        details.append({
            "rank": i + 1,
            "doc_id": source_id,
            "chunk_id": doc_id,
            "score": score,
            "data_type": data_type,
            "title": title,
            "content": content[:500],
            "metadata": meta,
        })

    results_md = "\n".join(lines)
    metrics_info = f"**Latency**: {latency_ms:.2f}ms | **Top {top_k}**: 판례 {precedent_count}, 법령 {law_count}"

    import json
    details_json = json.dumps(details, ensure_ascii=False, indent=2)

    return results_md, metrics_info, details_json


async def get_document_detail(doc_id: str) -> str:
    """문서 상세 정보 조회"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from sqlalchemy import select
    from app.common.database import async_session_factory
    from app.models.precedent_document import PrecedentDocument
    from app.models.law_document import LawDocument

    async with async_session_factory() as session:
        precedent_result = await session.execute(
            select(PrecedentDocument).where(
                PrecedentDocument.serial_number == doc_id
            )
        )
        precedent = precedent_result.scalar_one_or_none()

        if precedent:
            return f"""## 판례 상세: {precedent.case_name}

**사건번호**: {precedent.case_number}
**법원**: {precedent.court_name}
**선고일**: {precedent.decision_date}

### 판시사항
{precedent.summary or "없음"}

### 판결요지
{precedent.reasoning or "없음"}

### 주문
{precedent.ruling or "없음"}

### 참조조문
{precedent.reference_provisions or "없음"}
"""

        law_result = await session.execute(
            select(LawDocument).where(LawDocument.law_id == doc_id)
        )
        law = law_result.scalar_one_or_none()

        if law:
            return f"""## 법령 상세: {law.law_name}

**법령 ID**: {law.law_id}
**소관부처**: {law.ministry}
**시행일**: {law.enforcement_date}

### 조문 내용
{law.content or "없음"}

### 부칙
{law.supplementary or "없음"}
"""

        return f"문서를 찾을 수 없습니다: {doc_id}"


def add_search_result_to_ground_truth(doc_ids: str) -> str:
    """검색 결과를 Ground Truth에 추가"""
    from evaluation.ui.dataset_editor import add_to_ground_truth
    _, status = add_to_ground_truth(doc_ids)
    return status


def create_search_analyzer_tab():
    """검색 분석 탭 생성"""
    with gr.Column():
        gr.Markdown("## 실시간 벡터 검색 테스트")

        with gr.Row():
            query_input = gr.Textbox(
                label="검색 쿼리",
                placeholder="손해배상 청구 요건",
                scale=4,
            )
            top_k_input = gr.Slider(
                minimum=5,
                maximum=50,
                value=10,
                step=5,
                label="Top K",
                scale=1,
            )
            doc_type_filter = gr.Dropdown(
                choices=["all", "판례", "법령"],
                value="all",
                label="문서 유형",
                scale=1,
            )
            search_btn = gr.Button("검색 실행", variant="primary", scale=1)

        metrics_display = gr.Markdown(label="메트릭")

        with gr.Tabs():
            with gr.TabItem("검색 결과"):
                search_results = gr.Markdown(
                    value="쿼리를 입력하고 검색 버튼을 누르세요.",
                )

            with gr.TabItem("상세 JSON"):
                results_json = gr.Code(
                    language="json",
                    label="검색 결과 상세",
                )

        gr.Markdown("---")
        gr.Markdown("## 문서 상세 보기")

        with gr.Row():
            detail_doc_id = gr.Textbox(
                label="문서 ID",
                placeholder="76396",
                scale=3,
            )
            detail_btn = gr.Button("상세 보기", scale=1)
            add_gt_btn = gr.Button("Ground Truth에 추가", scale=1)

        document_detail = gr.Markdown(label="문서 상세")
        add_gt_status = gr.Textbox(label="추가 상태", interactive=False)

    search_btn.click(
        fn=search_vectors,
        inputs=[query_input, top_k_input, doc_type_filter],
        outputs=[search_results, metrics_display, results_json],
    )

    detail_btn.click(
        fn=get_document_detail,
        inputs=[detail_doc_id],
        outputs=[document_detail],
    )

    add_gt_btn.click(
        fn=add_search_result_to_ground_truth,
        inputs=[detail_doc_id],
        outputs=[add_gt_status],
    )
