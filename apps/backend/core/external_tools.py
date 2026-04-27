from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


async def check_payment_status(*, query: str) -> dict:
    """Stub billing tool. Replace with real billing API call."""
    q = (query or "").lower()
    status = "unknown"
    if any(k in q for k in ("не работает", "ошибка", "списали", "оплат")):
        status = "needs_manual_check"
    return {
        "status": status,
        "query": query,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


async def create_crm_ticket(*, reason: str, customer_text: str) -> dict:
    """Stub CRM tool. Replace with real CRM integration."""
    return {
        "ticket_id": str(uuid4()),
        "reason": reason[:256],
        "customer_text": customer_text[:512],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
