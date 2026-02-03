"""
데이터셋 빌더 UI (역추적 방식)

1. 문서 검색 (PostgreSQL + LanceDB)
2. Ground Truth 문서 선택 (체크박스)
3. 질문 작성 폼
4. 메타데이터 입력 (카테고리, 유형, 난이도)
5. 데이터셋 저장/내보내기
"""

import json
from pathlib import Path
from typing import Optional

import gradio as gr

from evaluation.schemas import (
    Category,
    QueryType,
    Difficulty,
    DocumentType,
)
from evaluation.tools.dataset_builder import DatasetBuilder
from evaluation.config import eval_settings


_current_dataset: Optional[DatasetBuilder] = None
_selected_documents: list[dict] = []


def get_or_create_dataset(name: str = "eval_dataset_v1") -> DatasetBuilder:
    """데이터셋 가져오기 또는 생성"""
    global _current_dataset

    if _current_dataset is None:
        dataset_path = eval_settings.datasets_dir / f"{name}.json"
        if dataset_path.exists():
            _current_dataset = DatasetBuilder.load(dataset_path)
        else:
            _current_dataset = DatasetBuilder(
                name=name,
                description="RAG 평가 데이터셋",
            )

    return _current_dataset


async def search_documents(
    query: str,
    doc_type: str,
    limit: int = 20,
) -> list[dict]:
    """
    문서 검색 (PostgreSQL + LanceDB)

    PostgreSQL에서 키워드 검색 후 결과 반환
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from sqlalchemy import select, or_
    from app.core.database import async_session_factory
    from app.models.precedent_document import PrecedentDocument
    from app.models.law_document import LawDocument

    results = []

    async with async_session_factory() as session:
        if doc_type in ("판례", "precedent", "all"):
            stmt = (
                select(PrecedentDocument)
                .where(
                    or_(
                        PrecedentDocument.case_name.ilike(f"%{query}%"),
                        PrecedentDocument.summary.ilike(f"%{query}%"),
                        PrecedentDocument.case_number.ilike(f"%{query}%"),
                    )
                )
                .limit(limit)
            )
            result = await session.execute(stmt)
            for p in result.scalars():
                results.append({
                    "doc_id": p.serial_number,
                    "doc_type": "precedent",
                    "title": f"[판례] {p.case_name or p.case_number}",
                    "subtitle": f"{p.court_name} | {p.decision_date}",
                    "content": (p.summary or "")[:300],
                    "full_content": p.summary,
                })

        if doc_type in ("법령", "law", "all"):
            stmt = (
                select(LawDocument)
                .where(
                    or_(
                        LawDocument.law_name.ilike(f"%{query}%"),
                        LawDocument.content.ilike(f"%{query}%"),
                    )
                )
                .limit(limit)
            )
            result = await session.execute(stmt)
            for l in result.scalars():
                results.append({
                    "doc_id": l.law_id,
                    "doc_type": "law",
                    "title": f"[법령] {l.law_name}",
                    "subtitle": f"{l.ministry} | {l.enforcement_date}",
                    "content": (l.content or "")[:300],
                    "full_content": l.content,
                })

    return results


def format_search_results(results: list[dict]) -> str:
    """검색 결과 포맷팅"""
    if not results:
        return "검색 결과가 없습니다."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"### {i}. {r['title']}")
        lines.append(f"**{r['subtitle']}**")
        lines.append(f"```\n{r['content']}...\n```")
        lines.append(f"*ID: {r['doc_id']}*")
        lines.append("")

    return "\n".join(lines)


def add_to_ground_truth(doc_ids: str) -> tuple[str, str]:
    """문서를 Ground Truth에 추가"""
    global _selected_documents

    ids = [id.strip() for id in doc_ids.split(",") if id.strip()]
    added = []

    for doc_id in ids:
        if not any(d["doc_id"] == doc_id for d in _selected_documents):
            doc_type = "precedent" if doc_id.isdigit() else "law"
            _selected_documents.append({
                "doc_id": doc_id,
                "doc_type": doc_type,
            })
            added.append(doc_id)

    selected_text = "\n".join([
        f"- {d['doc_id']} ({d['doc_type']})"
        for d in _selected_documents
    ]) or "선택된 문서 없음"

    return selected_text, f"추가됨: {', '.join(added)}" if added else "이미 선택된 문서입니다."


def clear_selected_documents() -> str:
    """선택된 문서 초기화"""
    global _selected_documents
    _selected_documents = []
    return "선택된 문서 없음"


def save_query(
    question: str,
    category: str,
    query_type: str,
    difficulty: str,
    key_points: str,
    citations: str,
) -> str:
    """쿼리 저장"""
    global _selected_documents

    if not question.strip():
        return "⚠️ 질문을 입력해주세요."

    if not _selected_documents:
        return "⚠️ Ground Truth 문서를 선택해주세요."

    dataset = get_or_create_dataset()

    key_points_list = [kp.strip() for kp in key_points.split("\n") if kp.strip()]
    citations_list = [c.strip() for c in citations.split("\n") if c.strip()]

    try:
        query = dataset.add_query(
            question=question,
            source_documents=_selected_documents.copy(),
            category=category,
            query_type=query_type,
            difficulty=difficulty,
            key_points=key_points_list,
            required_citations=citations_list,
            source="manual",
        )

        _selected_documents.clear()
        dataset.save()

        return f"✅ 저장됨: {query.id}\n총 {len(dataset.dataset.queries)}개 쿼리"

    except Exception as e:
        return f"❌ 저장 실패: {str(e)}"


def get_dataset_summary() -> str:
    """데이터셋 요약"""
    dataset = get_or_create_dataset()
    stats = dataset.get_statistics()

    if stats["total"] == 0:
        return "데이터셋이 비어있습니다."

    lines = [
        f"## 데이터셋: {dataset.dataset.name}",
        f"**총 쿼리 수**: {stats['total']}",
        "",
        "### 카테고리별",
    ]

    for cat, count in stats.get("by_category", {}).items():
        lines.append(f"- {cat}: {count}")

    lines.extend(["", "### 쿼리 유형별"])
    for qtype, count in stats.get("by_type", {}).items():
        lines.append(f"- {qtype}: {count}")

    lines.extend(["", "### 생성 방식별"])
    for src, count in stats.get("by_source", {}).items():
        lines.append(f"- {src}: {count}")

    return "\n".join(lines)


def get_dataset_queries() -> str:
    """데이터셋 쿼리 목록"""
    dataset = get_or_create_dataset()

    if not dataset.dataset.queries:
        return "쿼리가 없습니다."

    lines = []
    for q in dataset.dataset.queries:
        gt_docs = ", ".join([d.doc_id for d in q.ground_truth.source_documents])
        lines.append(f"### {q.id}")
        lines.append(f"**질문**: {q.question}")
        lines.append(f"**카테고리**: {q.metadata.category.value} | **유형**: {q.metadata.query_type.value}")
        lines.append(f"**Ground Truth**: {gt_docs}")
        lines.append("")

    return "\n".join(lines)


def export_dataset() -> str:
    """데이터셋 내보내기"""
    dataset = get_or_create_dataset()
    path = dataset.save()
    return f"저장됨: {path}"


def create_dataset_editor_tab():
    """데이터셋 빌더 탭 생성"""
    with gr.Column():
        gr.Markdown("## 1. 문서 검색 (Ground Truth 선택용)")

        with gr.Row():
            search_input = gr.Textbox(
                label="검색어",
                placeholder="손해배상, 민법, 대법원 등",
                scale=3,
            )
            doc_type_select = gr.Dropdown(
                choices=["all", "판례", "법령"],
                value="all",
                label="문서 유형",
                scale=1,
            )
            search_btn = gr.Button("검색", variant="primary", scale=1)

        search_results = gr.Markdown(
            value="검색어를 입력하고 검색 버튼을 누르세요.",
            label="검색 결과",
        )

        gr.Markdown("---")
        gr.Markdown("## 2. Ground Truth 문서 선택")

        with gr.Row():
            doc_id_input = gr.Textbox(
                label="문서 ID (쉼표로 구분)",
                placeholder="76396, 010719",
                scale=3,
            )
            add_doc_btn = gr.Button("추가", scale=1)
            clear_doc_btn = gr.Button("초기화", scale=1)

        selected_docs_display = gr.Markdown(
            value="선택된 문서 없음",
            label="선택된 문서",
        )
        add_status = gr.Textbox(label="상태", interactive=False)

        gr.Markdown("---")
        gr.Markdown("## 3. 질문 작성")

        question_input = gr.Textbox(
            label="질문",
            placeholder="임대차 보증금 반환 청구 요건은 무엇인가요?",
            lines=2,
        )

        with gr.Row():
            category_select = gr.Dropdown(
                choices=[c.value for c in Category],
                value=Category.CIVIL.value,
                label="카테고리",
            )
            query_type_select = gr.Dropdown(
                choices=[qt.value for qt in QueryType],
                value=QueryType.CONCEPT_SEARCH.value,
                label="쿼리 유형",
            )
            difficulty_select = gr.Dropdown(
                choices=[d.value for d in Difficulty],
                value=Difficulty.MEDIUM.value,
                label="난이도",
            )

        with gr.Row():
            key_points_input = gr.Textbox(
                label="Key Points (줄바꿈으로 구분)",
                placeholder="계약 종료 요건\n동시이행 관계",
                lines=3,
            )
            citations_input = gr.Textbox(
                label="Required Citations (줄바꿈으로 구분)",
                placeholder="민법 제621조",
                lines=3,
            )

        with gr.Row():
            save_btn = gr.Button("데이터셋에 추가", variant="primary")
            save_status = gr.Textbox(label="저장 상태", interactive=False)

        gr.Markdown("---")
        gr.Markdown("## 4. 현재 데이터셋")

        with gr.Row():
            refresh_btn = gr.Button("새로고침")
            export_btn = gr.Button("내보내기")

        with gr.Tabs():
            with gr.TabItem("요약"):
                dataset_summary = gr.Markdown(value=get_dataset_summary())

            with gr.TabItem("전체 쿼리"):
                dataset_queries = gr.Markdown(value=get_dataset_queries())

        export_status = gr.Textbox(label="내보내기 상태", interactive=False)

    async def do_search(query: str, doc_type: str) -> str:
        if not query.strip():
            return "검색어를 입력해주세요."
        results = await search_documents(query, doc_type)
        return format_search_results(results)

    search_btn.click(
        fn=do_search,
        inputs=[search_input, doc_type_select],
        outputs=[search_results],
    )

    add_doc_btn.click(
        fn=add_to_ground_truth,
        inputs=[doc_id_input],
        outputs=[selected_docs_display, add_status],
    )

    clear_doc_btn.click(
        fn=clear_selected_documents,
        outputs=[selected_docs_display],
    )

    save_btn.click(
        fn=save_query,
        inputs=[
            question_input,
            category_select,
            query_type_select,
            difficulty_select,
            key_points_input,
            citations_input,
        ],
        outputs=[save_status],
    )

    def refresh_dataset() -> tuple[str, str]:
        return get_dataset_summary(), get_dataset_queries()

    refresh_btn.click(
        fn=refresh_dataset,
        outputs=[dataset_summary, dataset_queries],
    )

    export_btn.click(
        fn=export_dataset,
        outputs=[export_status],
    )
