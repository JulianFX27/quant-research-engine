from __future__ import annotations

from src.live.state.idempotency_store import IdempotencyStore, IdempotencyRecord


def main():
    store = IdempotencyStore("results/live_state/idempotency.sqlite")

    rec = IdempotencyRecord(
        idempotency_key="TEST_KEY_001",
        intent_id="INTENT_TEST_001",
        client_order_id="COID-TEST-001",
        state="READY",
        attempt_no=1,
    )
    store.upsert_new(rec)

    row = store.get("TEST_KEY_001")
    print("GET:", row)

    store.mark_state("TEST_KEY_001", state="ACCEPTED", broker_order_id=123456)
    row2 = store.get("TEST_KEY_001")
    print("UPDATED:", row2)

    print("RECENT:", store.list_recent(limit=5))


if __name__ == "__main__":
    main()