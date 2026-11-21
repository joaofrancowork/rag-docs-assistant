from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class EvalItem(BaseModel):
    """One labelled question. ``source`` and ``page`` are null for unanswerable ones."""

    id: str
    question: str
    source: str | None
    page: int | None
    must_include: list[str]

    @property
    def answerable(self) -> bool:
        return self.source is not None


def load_dataset(path: Path) -> list[EvalItem]:
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            items.append(EvalItem.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_no}: invalid eval item") from exc
    return items
