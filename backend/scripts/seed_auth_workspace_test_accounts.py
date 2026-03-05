#!/usr/bin/env python3
"""인증/워크스페이스 테스트 계정 시드 스크립트.

기본으로 3개 테스트 계정을 생성하고, 각 계정에 대해
워크스페이스 사건/타임라인/대화 샘플 데이터를 함께 생성한다.

Usage:
    uv run python scripts/seed_auth_workspace_test_accounts.py
    uv run python scripts/seed_auth_workspace_test_accounts.py --reset
    uv run python scripts/seed_auth_workspace_test_accounts.py --users-only
"""

from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.orm import Session

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.chat_conversation import ChatConversation, ChatMessage
from app.models.user import RefreshToken, SocialAccount, User
from app.models.workspace_case import (
    IdentityLink,
    WorkspaceActivityLog,
    WorkspaceCase,
    WorkspaceCaseTimelineItem,
)
from app.modules.auth.service import AuthService
from scripts.common.db import create_sync_session_factory
from scripts.common.logging_config import setup_logging

logger = setup_logging(__name__)

SEED_PROVIDER = "seed"


@dataclass(frozen=True)
class SeedAccount:
    email: str
    password: str
    display_name: str
    role: str
    session_token: str
    case_name: str
    case_type: str


SEED_ACCOUNTS: tuple[SeedAccount, ...] = (
    SeedAccount(
        email="test.user1@law-platform.local",
        password="Test1234!",
        display_name="테스트 사용자 1",
        role="user",
        session_token="a1" * 32,
        case_name="임대차 분쟁 테스트 사건",
        case_type="civil",
    ),
    SeedAccount(
        email="test.user2@law-platform.local",
        password="Test1234!",
        display_name="테스트 사용자 2",
        role="user",
        session_token="b2" * 32,
        case_name="교통사고 합의 테스트 사건",
        case_type="civil",
    ),
    SeedAccount(
        email="test.lawyer@law-platform.local",
        password="Test1234!",
        display_name="테스트 변호사",
        role="lawyer",
        session_token="c3" * 32,
        case_name="형사 고소 대응 테스트 사건",
        case_type="criminal",
    ),
)


def _seed_tagged_items(case_type: str, actor_name: str) -> list[dict[str, object]]:
    """워크스페이스 샘플 태그 생성."""
    return [
        {
            "label": "사건유형",
            "value": case_type,
            "category": "case_type",
        },
        {
            "label": "당사자",
            "value": actor_name,
            "category": "party",
        },
    ]


def _seed_summary(case_name: str) -> dict[str, object]:
    """워크스페이스 샘플 요약 생성."""
    return {
        "overview": f"{case_name} 관련 초기 검토용 테스트 요약입니다.",
        "status": "draft",
        "next_steps": [
            "증빙자료 목록 정리",
            "사실관계 타임라인 점검",
        ],
    }


def _ensure_user(db: Session, account: SeedAccount) -> tuple[User, bool]:
    """사용자 계정을 생성/갱신."""
    result = db.execute(select(User).where(User.email == account.email))
    user = result.scalar_one_or_none()
    hashed = AuthService.hash_password(account.password)

    if user is None:
        user = User(
            email=account.email,
            hashed_password=hashed,
            display_name=account.display_name,
            role=account.role,
            email_verified=True,
            is_active=True,
        )
        db.add(user)
        db.flush()
        return user, True

    user.hashed_password = hashed
    user.display_name = account.display_name
    user.role = account.role
    user.email_verified = True
    user.is_active = True
    return user, False


def _ensure_identity_link(db: Session, user_id: uuid.UUID, session_token: str) -> None:
    """세션 토큰을 사용자와 연결."""
    result = db.execute(
        select(IdentityLink).where(
            IdentityLink.session_token == session_token,
            IdentityLink.provider == SEED_PROVIDER,
        )
    )
    link = result.scalar_one_or_none()

    if link is None:
        db.add(
            IdentityLink(
                session_token=session_token,
                user_id=user_id,
                provider=SEED_PROVIDER,
            )
        )
        return

    link.user_id = user_id


def _ensure_workspace_case(
    db: Session, account: SeedAccount
) -> tuple[WorkspaceCase, bool]:
    """워크스페이스 사건 생성/갱신."""
    result = db.execute(
        select(WorkspaceCase).where(
            WorkspaceCase.session_token == account.session_token,
            WorkspaceCase.case_name == account.case_name,
        )
    )
    case = result.scalar_one_or_none()

    tagged_items = _seed_tagged_items(account.case_type, account.display_name)
    summary = _seed_summary(account.case_name)

    if case is None:
        case = WorkspaceCase(
            session_token=account.session_token,
            case_name=account.case_name,
            case_type=account.case_type,
            status="active",
            tagged_items=tagged_items,
            summary=summary,
        )
        db.add(case)
        db.flush()
        return case, True

    case.case_type = account.case_type
    case.status = "active"
    case.tagged_items = tagged_items
    case.summary = summary
    return case, False


def _ensure_timeline_items(db: Session, case_id: uuid.UUID) -> int:
    """타임라인 항목이 없으면 기본 2개 생성."""
    existing_count = db.execute(
        select(func.count()).where(WorkspaceCaseTimelineItem.case_id == case_id)
    ).scalar_one()
    if existing_count > 0:
        return 0

    db.add_all(
        [
            WorkspaceCaseTimelineItem(
                case_id=case_id,
                date_text="2026-01-12",
                title="사건 접수",
                description="테스트 사건이 워크스페이스에 등록되었습니다.",
                category="접수",
                source_type="manual",
                sort_order=1,
            ),
            WorkspaceCaseTimelineItem(
                case_id=case_id,
                date_text="2026-01-15",
                title="초기 상담 기록",
                description="핵심 사실관계와 요청사항을 정리했습니다.",
                category="상담",
                source_type="manual",
                sort_order=2,
            ),
        ]
    )
    return 2


def _thread_id_for(account: SeedAccount) -> str:
    email_prefix = account.email.split("@", 1)[0].replace(".", "-")
    return f"seed-{email_prefix}-thread"


def _ensure_conversation(
    db: Session,
    account: SeedAccount,
    case_id: uuid.UUID,
) -> tuple[uuid.UUID, int]:
    """사건 연결용 샘플 대화 생성/갱신."""
    thread_id = _thread_id_for(account)
    result = db.execute(
        select(ChatConversation).where(ChatConversation.thread_id == thread_id)
    )
    conversation = result.scalar_one_or_none()

    if conversation is None:
        conversation = ChatConversation(
            thread_id=thread_id,
            session_token=account.session_token,
            case_id=case_id,
            title=f"{account.case_name} 상담",
            case_type=account.case_type,
            last_agent="workspace",
            tagged_items=_seed_tagged_items(account.case_type, account.display_name),
        )
        db.add(conversation)
        db.flush()
    else:
        conversation.session_token = account.session_token
        conversation.case_id = case_id
        conversation.title = f"{account.case_name} 상담"
        conversation.case_type = account.case_type
        conversation.last_agent = "workspace"
        conversation.tagged_items = _seed_tagged_items(
            account.case_type, account.display_name
        )

    message_count = db.execute(
        select(func.count()).where(ChatMessage.conversation_id == conversation.id)
    ).scalar_one()
    if message_count > 0:
        return conversation.id, 0

    db.add_all(
        [
            ChatMessage(
                conversation_id=conversation.id,
                role="user",
                content=f"{account.case_name} 관련으로 초기 상담을 요청합니다.",
                agent_type="workspace",
            ),
            ChatMessage(
                conversation_id=conversation.id,
                role="assistant",
                content="사실관계, 증거, 일정 정보를 우선 정리해보겠습니다.",
                agent_type="workspace",
            ),
        ]
    )
    return conversation.id, 2


def _reset_seed_data(db: Session) -> None:
    """기존 시드 데이터 삭제."""
    emails = [account.email for account in SEED_ACCOUNTS]
    session_tokens = [account.session_token for account in SEED_ACCOUNTS]

    user_ids = list(
        db.execute(select(User.id).where(User.email.in_(emails))).scalars().all()
    )

    conversation_ids = list(
        db.execute(
            select(ChatConversation.id).where(
                ChatConversation.session_token.in_(session_tokens)
            )
        )
        .scalars()
        .all()
    )
    if conversation_ids:
        db.execute(
            delete(ChatMessage).where(ChatMessage.conversation_id.in_(conversation_ids))
        )
        db.execute(
            delete(ChatConversation).where(ChatConversation.id.in_(conversation_ids))
        )

    case_ids = list(
        db.execute(
            select(WorkspaceCase.id).where(
                WorkspaceCase.session_token.in_(session_tokens)
            )
        )
        .scalars()
        .all()
    )
    if case_ids:
        db.execute(
            delete(WorkspaceCaseTimelineItem).where(
                WorkspaceCaseTimelineItem.case_id.in_(case_ids)
            )
        )
        db.execute(delete(WorkspaceCase).where(WorkspaceCase.id.in_(case_ids)))

    db.execute(
        delete(WorkspaceActivityLog).where(
            WorkspaceActivityLog.session_token.in_(session_tokens)
        )
    )

    identity_link_filter = IdentityLink.session_token.in_(session_tokens)
    if user_ids:
        identity_link_filter = or_(
            identity_link_filter,
            and_(
                IdentityLink.user_id.in_(user_ids),
                IdentityLink.provider == SEED_PROVIDER,
            ),
        )
    db.execute(delete(IdentityLink).where(identity_link_filter))

    if user_ids:
        db.execute(delete(RefreshToken).where(RefreshToken.user_id.in_(user_ids)))
        db.execute(delete(SocialAccount).where(SocialAccount.user_id.in_(user_ids)))

    db.execute(delete(User).where(User.email.in_(emails)))


def _print_credentials() -> None:
    print("\n테스트 로그인 계정")
    print("-" * 60)
    for account in SEED_ACCOUNTS:
        print(
            f"email={account.email} | password={account.password} | role={account.role}"
        )
    print("-" * 60)
    print("모든 계정의 비밀번호는 동일합니다.")


def seed_accounts(reset: bool, users_only: bool) -> None:
    """계정/워크스페이스 시드 실행."""
    session_factory = create_sync_session_factory()

    with session_factory() as db:
        if reset:
            _reset_seed_data(db)
            db.commit()
            logger.info("기존 테스트 계정/워크스페이스 데이터를 삭제했습니다.")

        created_users = 0
        created_cases = 0
        created_timeline_items = 0
        created_messages = 0

        for account in SEED_ACCOUNTS:
            user, user_created = _ensure_user(db, account)
            if user_created:
                created_users += 1
            _ensure_identity_link(db, user.id, account.session_token)

            if users_only:
                logger.info("계정 준비: %s", account.email)
                continue

            case, case_created = _ensure_workspace_case(db, account)
            if case_created:
                created_cases += 1

            created_timeline_items += _ensure_timeline_items(db, case.id)
            _, added_messages = _ensure_conversation(db, account, case.id)
            created_messages += added_messages

            logger.info("계정/사건 준비: %s / %s", account.email, account.case_name)

        db.commit()

    logger.info(
        "완료: users=%d, new_cases=%d, new_timeline_items=%d, new_messages=%d",
        created_users,
        created_cases,
        created_timeline_items,
        created_messages,
    )
    _print_credentials()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="테스트 계정 + 워크스페이스 샘플 데이터 생성",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 시드 데이터를 삭제한 뒤 다시 생성",
    )
    parser.add_argument(
        "--users-only",
        action="store_true",
        help="로그인 테스트용 사용자 계정만 생성 (워크스페이스 데이터 미생성)",
    )
    args = parser.parse_args()

    seed_accounts(reset=args.reset, users_only=args.users_only)


if __name__ == "__main__":
    main()
