from abc import ABC, abstractmethod


class BaseKMeans(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, iterations: int):
        pass

    @abstractmethod
    def predict(self, X):
        pass
