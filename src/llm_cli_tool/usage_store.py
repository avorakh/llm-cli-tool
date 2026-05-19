import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class UsageEvent:
    session_id: str
    created_at: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_sec: float
    streamed: bool
    chat_turn: bool
    request_mode: str


@dataclass(frozen=True)
class WeeklyCostSummary:
    total_cost_usd: float
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    top_model: str | None
    top_model_cost_usd: float

    @property
    def top_model_share(self) -> float:
        if self.total_cost_usd == 0:
            return 0.0
        return (self.top_model_cost_usd / self.total_cost_usd) * 100


class UsageStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or Path.home() / ".llm-cli-tool" / "usage.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def log_usage(self, event: UsageEvent) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO usage_events (
                    session_id,
                    created_at,
                    model,
                    provider,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    latency_sec,
                    streamed,
                    chat_turn,
                    request_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.session_id,
                    event.created_at,
                    event.model,
                    event.provider,
                    event.input_tokens,
                    event.output_tokens,
                    event.cost_usd,
                    event.latency_sec,
                    int(event.streamed),
                    int(event.chat_turn),
                    event.request_mode,
                ),
            )

    def get_weekly_summary(self, now: datetime | None = None) -> WeeklyCostSummary:
        local_now = now or datetime.now().astimezone()
        week_start_local = (local_now - timedelta(days=local_now.weekday())).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        week_start_utc = week_start_local.astimezone(timezone.utc).isoformat()

        with sqlite3.connect(self.db_path) as connection:
            totals = connection.execute(
                """
                SELECT
                    COALESCE(SUM(cost_usd), 0),
                    COUNT(*),
                    COALESCE(SUM(input_tokens), 0),
                    COALESCE(SUM(output_tokens), 0)
                FROM usage_events
                WHERE created_at >= ?
                """,
                (week_start_utc,),
            ).fetchone()

            top_model_row = connection.execute(
                """
                SELECT model, SUM(cost_usd) AS model_cost
                FROM usage_events
                WHERE created_at >= ?
                GROUP BY model
                ORDER BY model_cost DESC
                LIMIT 1
                """,
                (week_start_utc,),
            ).fetchone()

        top_model = None
        top_model_cost = 0.0
        if top_model_row is not None:
            top_model = top_model_row[0]
            top_model_cost = float(top_model_row[1] or 0.0)

        return WeeklyCostSummary(
            total_cost_usd=float(totals[0] or 0.0),
            total_calls=int(totals[1] or 0),
            total_input_tokens=int(totals[2] or 0),
            total_output_tokens=int(totals[3] or 0),
            top_model=top_model,
            top_model_cost_usd=top_model_cost,
        )

    def _initialize(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    latency_sec REAL NOT NULL,
                    streamed INTEGER NOT NULL,
                    chat_turn INTEGER NOT NULL,
                    request_mode TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_usage_events_created_at
                ON usage_events (created_at)
                """
            )
