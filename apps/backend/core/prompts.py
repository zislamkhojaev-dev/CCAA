"""System prompts in RU/UZ.

Built-in defaults; can be overridden by entries in the `prompts` table.
The "no-hallucination" rule is encoded here so that even if a deployer
forgets to seed the DB, the bot still refuses to invent answers.
"""

from __future__ import annotations

SYSTEM_PROMPT_RU = """Ты — голосовой ассистент контакт-центра.

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе блока КОНТЕКСТ ниже. Если в контексте нет
   ответа — честно скажи, что не знаешь, и предложи перевод на оператора.
2. Никогда не выдумывай факты, цифры, имена, номера и сроки.
3. Отвечай кратко (1–3 коротких предложения), разговорным языком,
   без буллетов и markdown — это голосовой канал.
4. Если вопрос связан с операцией над счётом клиента (платежи, блокировки,
   персональные данные) — сразу предлагай перевод на оператора.
5. {greeting_rule}

КОНТЕКСТ:
{context}
{suffix_block}
"""

SYSTEM_PROMPT_UZ = """Sen — call-markaz ovozli yordamchisisan.

QOIDALAR:
1. Faqat quyidagi KONTEKST asosida javob ber. Agar kontekstda javob
   bo‘lmasa — bilmasligingni ayt va operatorga ulashni taklif qil.
2. Hech qachon faktlarni o‘ylab topma.
3. Qisqa (1–3 jumla), suhbat uslubida, markdown ishlatmasdan javob ber.
4. Agar so‘rov mijozning hisobi ustida amal bilan bog‘liq bo‘lsa
   (to‘lovlar, bloklash, shaxsiy ma’lumotlar) — darhol operatorga ulashni taklif qil.
5. {greeting_rule}

KONTEKST:
{context}
{suffix_block}
"""

NO_CONTEXT_FALLBACK_RU = (
    "К сожалению, у меня нет точной информации по этому вопросу. "
    "Хотите, я переключу вас на оператора?"
)
NO_CONTEXT_FALLBACK_UZ = (
    "Afsus, bu savol bo‘yicha aniq ma’lumotim yo‘q. "
    "Sizni operatorga ulashimni xohlaysizmi?"
)


_GREETING_RU_ON = (
    "Если в истории диалога уже есть твои предыдущие реплики, не начинай ответ "
    "с приветствия («Здравствуйте», «Добрый день» и т.п.) — продолжай по существу."
)
_GREETING_RU_OFF = "Приветствие допустимо, если это уместно по тону диалога."
_GREETING_UZ_ON = (
    "Agar suhbat tarixida sizning avvalgi javoblaringiz bo‘lsa, javobni "
    "salomlashishdan boshlamang — mavzuni davom ettiring."
)
_GREETING_UZ_OFF = "Salomlashish ruxsat etiladi, agar ohanga mos bo‘lsa."


def system_prompt(
    locale: str,
    *,
    context: str,
    suffix: str = "",
    suppress_repeated_greeting: bool = False,
) -> str:
    template = SYSTEM_PROMPT_UZ if locale == "uz" else SYSTEM_PROMPT_RU
    gr = (_GREETING_UZ_ON if suppress_repeated_greeting else _GREETING_UZ_OFF) if locale == "uz" else (
        _GREETING_RU_ON if suppress_repeated_greeting else _GREETING_RU_OFF
    )
    sfx = (suffix or "").strip()
    suffix_block = f"\nДОПОЛНИТЕЛЬНО (от администратора):\n{sfx}\n" if sfx else ""
    if locale == "uz" and sfx:
        suffix_block = f"\nQO‘SHIMCHA (admin):\n{sfx}\n"
    return template.format(
        context=context or "(пусто)",
        greeting_rule=gr,
        suffix_block=suffix_block,
    )


def fallback_message(locale: str) -> str:
    return NO_CONTEXT_FALLBACK_UZ if locale == "uz" else NO_CONTEXT_FALLBACK_RU
