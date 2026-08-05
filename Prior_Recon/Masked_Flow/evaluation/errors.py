from __future__ import annotations


class EvaluationInputError(ValueError):
    def __init__(self, field: str, detail: str) -> None:
        self.field = field
        self.detail = detail
        super().__init__(str(self))

    def __str__(self) -> str:
        return f"invalid {self.field}: {self.detail}"
