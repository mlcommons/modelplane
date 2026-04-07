from typing import Any, Optional


class EvalContext:
    """Context state passed around during DAG execution."""

    def __init__(
        self, prompt: str, response: str, metadata: Optional[dict[str, Any]] = None
    ) -> None:
        self.prompt = prompt
        self.response = response
        self.metadata = metadata or {}
        self._parent_outputs = {}

    def set_parent_outputs(self, outputs: dict[str, Any]) -> None:
        self._parent_outputs = outputs

    def parent_outputs(self) -> list[Any]:
        """Return the NodeOutput for a specific node, or None if it was skipped."""
        return list(self._parent_outputs.values())
