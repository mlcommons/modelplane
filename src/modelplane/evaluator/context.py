from typing import Any


class EvalContext:
    """Context state passed around during DAG execution."""

    def __init__(self, prompt_text: str, response: str) -> None:
        self.prompt_text = prompt_text
        self.response = response
        self._parent_outputs = {}

    def set_parent_outputs(self, outputs: dict[str, Any]) -> None:
        self._parent_outputs = outputs

    def parent_outputs(self) -> list[Any]:
        """Return the NodeOutput for a specific node, or None if it was skipped."""
        return list(self._parent_outputs.values())
