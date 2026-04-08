from abc import abstractmethod


class Output:
    @property
    @abstractmethod
    def name(self) -> str:
        """Return a string name for this output, used for routing and debugging."""

    def __repr__(self) -> str:
        return f"{self.name} ({self.__class__.__name__})"
