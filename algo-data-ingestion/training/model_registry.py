from __future__ import annotations

from typing import Dict, Protocol, Type, TypeVar


class BaseModel(Protocol):
    def fit(self, X_train, y_train, **kwargs): ...
    def predict_proba(self, X): ...
    def save(self, path: str): ...

    @classmethod
    def load(cls, path: str): ...


T = TypeVar("T", bound=BaseModel)

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(name: str, cls: Type[T]) -> None:
    MODEL_REGISTRY[name] = cls


def get_model_class(name: str) -> Type[BaseModel]:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]
