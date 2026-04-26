"""Seed the database with demo prompts, voices, two KB pools, and analytics criteria.

Run inside the backend container:
    docker compose -f docker/docker-compose.yml exec backend \
        python -m apps.backend.scripts.seed
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from sqlalchemy import select

from apps.backend.config import get_settings
from apps.backend.models.db import init_db, session_scope
from apps.backend.models.entities import (
    AnalyticsCriterion,
    AnalyticsCriteriaSet,
    Document,
    Prompt,
    Voice,
)
from apps.backend.rag import get_rag_agent_assist, get_rag_voice
from apps.backend.utils.logging import configure_logging, get_logger

configure_logging()
log = get_logger("seed")


PROMPTS = [
    {
        "name": "default-ru",
        "locale": "ru",
        "content": (
            "Ты — голосовой ассистент банка. Отвечай только на основе контекста, "
            "коротко и разговорно. Не выдумывай факты."
        ),
        "is_active": True,
    },
    {
        "name": "default-uz",
        "locale": "uz",
        "content": (
            "Sen bank ovozli yordamchisisan. Faqat kontekst asosida qisqa va "
            "suhbat uslubida javob ber."
        ),
        "is_active": True,
    },
]

VOICES_OPENAI = [
    {
        "name": "Alloy (нейтральный)",
        "provider": "openai",
        "provider_voice_id": "alloy",
        "locale": "ru",
        "is_default": True,
    },
    {
        "name": "Nova (женский)",
        "provider": "openai",
        "provider_voice_id": "nova",
        "locale": "ru",
        "is_default": False,
    },
]

# Полная база для голосового бота (клиентский канал).
DEMO_DOCS_VOICE_RU = [
    {
        "title": "Тарифы по вкладам",
        "text": (
            "Вклад «Накопительный»: ставка 16% годовых, минимальная сумма 1 000 000 сум, "
            "срок от 6 до 24 месяцев, частичное снятие не предусмотрено, проценты "
            "выплачиваются в конце срока. Капитализация — ежемесячная.\n\n"
            "Вклад «Гибкий»: ставка 12% годовых, пополнение и частичное снятие без "
            "потери процентов. Минимальная сумма 500 000 сум.\n\n"
            "Вклад «Долгосрочный»: ставка 18% годовых при сроке от 36 месяцев, "
            "досрочное расторжение возможно, но проценты пересчитываются по ставке «До востребования» (1%)."
        ),
    },
    {
        "title": "Режим работы и контакты",
        "text": (
            "Контакт-центр работает круглосуточно, без выходных. "
            "Отделения банка работают с понедельника по пятницу с 9:00 до 18:00, "
            "в субботу с 10:00 до 14:00, воскресенье — выходной. "
            "Главный офис расположен в Ташкенте, проспект Амира Темура, 1. "
            "Бесплатный номер для звонков по Узбекистану: 1234."
        ),
    },
    {
        "title": "Комиссии и переводы",
        "text": (
            "Внутренние переводы между счетами клиента — бесплатно. "
            "Переводы клиентам нашего банка — 0%. "
            "Межбанковские переводы по Узбекистану — 0,3% от суммы, минимум 5 000 сум. "
            "Международные переводы SWIFT — 0,5% от суммы, минимум 25 USD."
        ),
    },
    {
        "title": "Карты и обслуживание",
        "text": (
            "Дебетовая карта «Старт» — выпуск бесплатный, годовое обслуживание 0 сум, "
            "лимит снятия 20 000 000 сум в сутки. "
            "Карта «Премиум» — выпуск бесплатный, годовое обслуживание 1 200 000 сум, "
            "включает страховку выезжающих за рубеж и доступ в бизнес-залы аэропортов. "
            "Срок выпуска любой карты — до 5 рабочих дней."
        ),
    },
]

# Узкая база для суфлёра: сценарии, скрипты, эскалации — не дублирует полные тарифы.
DEMO_DOCS_ASSIST_RU = [
    {
        "title": "Суфлёр: приветствие и прощание",
        "text": (
            "Приветствие: «Добрый день, меня зовут …, чем могу помочь?» "
            "Перед удержанием: «Пожалуйста, оставайтесь на линии, уточняю информацию». "
            "Завершение: «Спасибо за обращение, всего доброго»."
        ),
    },
    {
        "title": "Суфлёр: эскалация и риски",
        "text": (
            "При запросе на блокировку карты, перевод крупной суммы третьим лицам, "
            "жалобе на сотрудника или смене паспортных данных — оформите перевод на "
            "старшего специалиста, не давайте обещаний по срокам сами. "
            "Фраза: «Подключу коллегу, который детально проконсультирует»."
        ),
    },
    {
        "title": "Суфлёр: быстрые ответы без цифр",
        "text": (
            "Если клиент спрашивает точную ставку по вкладу — направьте в мобильное "
            "приложение или на голосового ассистента IVR; оператору не зачитывать "
            "длинные проценты с телефона. Коротко: «Актуальные условия видны в приложении "
            "в разделе Вклады»."
        ),
    },
]


async def upsert_prompts() -> None:
    async with session_scope() as s:
        for p in PROMPTS:
            existing = (
                await s.execute(select(Prompt).where(Prompt.name == p["name"]))
            ).scalar_one_or_none()
            if existing:
                continue
            s.add(Prompt(**p))
            log.info("prompt_seeded", name=p["name"])


async def upsert_voices() -> None:
    settings = get_settings()
    voices = VOICES_OPENAI if settings.tts_provider == "openai" else []
    if not voices:
        return
    async with session_scope() as s:
        for v in voices:
            existing = (
                await s.execute(
                    select(Voice).where(
                        Voice.provider == v["provider"],
                        Voice.provider_voice_id == v["provider_voice_id"],
                    )
                )
            ).scalar_one_or_none()
            if existing:
                continue
            s.add(Voice(**v))
            log.info("voice_seeded", name=v["name"])


async def upsert_documents_voice() -> None:
    rag = get_rag_voice()
    await rag.ensure_collection()
    async with session_scope() as s:
        for d in DEMO_DOCS_VOICE_RU:
            existing = (
                await s.execute(
                    select(Document).where(
                        Document.title == d["title"],
                        Document.knowledge_pool == "voice",
                    )
                )
            ).scalar_one_or_none()
            if existing:
                continue
            doc = Document(
                title=d["title"],
                locale="ru",
                source_type="seed",
                knowledge_pool="voice",
            )
            s.add(doc)
            await s.flush()
            chunks = await rag.index_document(
                document_id=doc.id, title=doc.title, text=d["text"], locale="ru"
            )
            doc.chunk_count = chunks
            log.info("document_seeded", pool="voice", title=d["title"], chunks=chunks)


async def upsert_documents_assist() -> None:
    rag = get_rag_agent_assist()
    await rag.ensure_collection()
    async with session_scope() as s:
        for d in DEMO_DOCS_ASSIST_RU:
            existing = (
                await s.execute(
                    select(Document).where(
                        Document.title == d["title"],
                        Document.knowledge_pool == "agent_assist",
                    )
                )
            ).scalar_one_or_none()
            if existing:
                continue
            doc = Document(
                title=d["title"],
                locale="ru",
                source_type="seed",
                knowledge_pool="agent_assist",
            )
            s.add(doc)
            await s.flush()
            chunks = await rag.index_document(
                document_id=doc.id, title=doc.title, text=d["text"], locale="ru"
            )
            doc.chunk_count = chunks
            log.info("document_seeded", pool="agent_assist", title=d["title"], chunks=chunks)


async def seed_analytics_criteria() -> None:
    async with session_scope() as s:
        existing = (
            await s.execute(
                select(AnalyticsCriteriaSet).where(AnalyticsCriteriaSet.name == "default-qa")
            )
        ).scalar_one_or_none()
        if existing:
            return
        st = AnalyticsCriteriaSet(
            id=uuid4(),
            name="default-qa",
            locale="ru",
            is_default=True,
        )
        s.add(st)
        await s.flush()
        criteria = [
            AnalyticsCriterion(
                id=uuid4(),
                set_id=st.id,
                code="greeting",
                title="Приветствие и представление",
                description="Есть ли вежливое приветствие и имя оператора.",
                weight=1.0,
                max_score=10.0,
                rubric="0 если грубо или без приветствия; 10 если тёплое приветствие и имя.",
            ),
            AnalyticsCriterion(
                id=uuid4(),
                set_id=st.id,
                code="clarity",
                title="Ясность формулировок",
                description="Понятны ли ответы клиенту без лишнего жаргона.",
                weight=1.5,
                max_score=10.0,
                rubric="Оцени простоту языка и структуру ответа.",
            ),
            AnalyticsCriterion(
                id=uuid4(),
                set_id=st.id,
                code="compliance",
                title="Соблюдение регламентов",
                description="Нет обещаний невыполнимого; эскалация при рисках.",
                weight=2.0,
                max_score=10.0,
                rubric="Снижай балл, если оператор гарантирует сроки/ставки без оговорок.",
            ),
            AnalyticsCriterion(
                id=uuid4(),
                set_id=st.id,
                code="empathy",
                title="Эмпатия",
                description="Признание эмоций клиента, извинения при проблеме.",
                weight=1.0,
                max_score=10.0,
                rubric="0 если игнор недовольства; 10 если есть сочувствие и решение.",
            ),
        ]
        for c in criteria:
            s.add(c)
        log.info("analytics_criteria_seeded", set_id=str(st.id), count=len(criteria))


async def main() -> None:
    log.info("seed_starting")
    await init_db()
    await upsert_prompts()
    await upsert_voices()
    await upsert_documents_voice()
    await upsert_documents_assist()
    await seed_analytics_criteria()
    log.info("seed_complete")


if __name__ == "__main__":
    asyncio.run(main())
