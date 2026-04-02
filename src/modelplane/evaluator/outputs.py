from abc import abstractmethod


class Output:
    @abstractmethod
    def is_safe(self) -> bool:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__class__.__name__


class Violating(Output):

    def is_safe(self) -> bool:
        return False


class NonViolating(Output):

    def is_safe(self) -> bool:
        return True


VIOLATING = Violating()
NONVIOLATING = NonViolating()
