from abc import abstractmethod


class Output:
    @property
    @abstractmethod
    def name(self) -> str:
        """Return a string name for this output, used for routing and debugging."""

    def __repr__(self) -> str:
        return f"{self.name} ({self.__class__.__name__})"


class Safety(Output):

    def __init__(self, is_safe: bool) -> None:
        self.is_safe = is_safe

    @property
    def name(self) -> str:
        return "SAFE" if self.is_safe else "UNSAFE"


SAFE = Safety(is_safe=True)
UNSAFE = Safety(is_safe=False)
