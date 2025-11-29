from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from training.infer import (
    ManifestArtifacts,
    apply_manifest_gates,
    load_manifest_artifacts,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """
    Declarative description of a deployable model manifest.

    Attributes:
        label: Logical name used by downstream systems (e.g. trading decisions).
        path: Filesystem path (directory) that contains the manifest bundle.
    """

    label: str
    path: Path


def parse_model_specs(raw_spec: Optional[str]) -> List[Tuple[str, str]]:
    """
    Parse a comma-delimited specification of deployable models.

    Supports the following syntaxes per entry:
        - "model_a"                     -> label: "model_a", relative path "model_a"
        - "alias=model_dir"             -> label: "alias",   relative path "model_dir"

    Whitespace around entries is ignored. Empty entries are skipped.
    """
    if not raw_spec:
        return []

    entries: List[Tuple[str, str]] = []
    for chunk in raw_spec.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "=" in part:
            label, rel_path = part.split("=", 1)
        else:
            label, rel_path = part, part
        label = label.strip()
        rel_path = rel_path.strip()
        if not label:
            raise ValueError(f"Invalid deployable model spec (missing label): {chunk!r}")
        if not rel_path:
            raise ValueError(f"Invalid deployable model spec (missing path): {chunk!r}")
        entries.append((label, rel_path))
    return entries


class ManifestRegistry:
    """
    Shared registry that caches manifest artifacts for deployable models.

    - Manifests are loaded once (typically during service startup) via `preload`.
    - Downstream scoring code calls `annotate_with_gate_pass` after probabilities
      are computed to attach the manifest gate decision to each row.
    - `prepare_decision_payload` materialises a minimal payload ready for trading.
    """

    def __init__(self) -> None:
        self._artifacts: Dict[str, ManifestArtifacts] = {}
        self._paths: Dict[str, Path] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------ #
    # Loading & lookup helpers
    # ------------------------------------------------------------------ #
    def preload(
        self,
        *,
        models_root: Path,
        specs: Sequence[Tuple[str, str]],
        clear: bool = False,
    ) -> List[ModelSpec]:
        """
        Load manifest artifacts for the provided specification list.

        Args:
            models_root: Base directory that relative paths will resolve against.
            specs: Iterable of (label, relative_or_absolute_path) tuples.
            clear: When True, the registry is cleared before loading.

        Returns:
            List[ModelSpec] describing successfully loaded models.
        """
        loaded: List[ModelSpec] = []
        if clear:
            with self._lock:
                self._artifacts.clear()
                self._paths.clear()

        for label, rel_path in specs:
            resolved = Path(rel_path)
            if not resolved.is_absolute():
                resolved = models_root / resolved
            resolved = resolved.expanduser().resolve()

            if not resolved.exists():
                raise FileNotFoundError(
                    f"Manifest directory for model '{label}' not found: {resolved}"
                )

            artifacts = load_manifest_artifacts(resolved, model_label=label)
            with self._lock:
                self._artifacts[label] = artifacts
                self._paths[label] = resolved
            loaded.append(ModelSpec(label=label, path=resolved))
            logger.info("Loaded manifest artifacts for model '%s' from %s", label, resolved)

        return loaded

    def ensure_loaded(self, label: str) -> ManifestArtifacts:
        """
        Retrieve cached artifacts. Raises KeyError if the model is unknown.
        """
        with self._lock:
            artifacts = self._artifacts.get(label)
        if artifacts is None:
            raise KeyError(f"Manifest artifacts not loaded for model '{label}'")
        return artifacts

    def get_probability_column(self, label: str) -> str:
        """
        Convenience accessor for the probability column declared in the manifest.
        """
        artifacts = self.ensure_loaded(label)
        return artifacts.prob_column

    def list_models(self) -> List[str]:
        """
        Return a sorted list of model labels currently cached.
        """
        with self._lock:
            return sorted(self._artifacts.keys())

    def get_path(self, label: str) -> Path:
        """
        Return the source directory associated with the given model label.
        """
        with self._lock:
            path = self._paths.get(label)
        if path is None:
            raise KeyError(f"Manifest path not tracked for model '{label}'")
        return path

    # ------------------------------------------------------------------ #
    # Gate helpers
    # ------------------------------------------------------------------ #
    def annotate_with_gate_pass(
        self,
        label: str,
        df: pd.DataFrame,
        *,
        prob_series: Optional[pd.Series] = None,
        gate_column: str = "gate_pass",
        mode: str = "inference",
        inplace: bool = False,
        update_metrics: bool = True,
    ) -> pd.DataFrame:
        """
        Apply the manifest gate predicates to the provided frame and attach the
        boolean decision as `gate_column`.

        Args:
            label: Model identifier registered in this registry.
            df: Feature/probability frame aligned to the manifest specification.
            prob_series: Optional explicit probability series; when omitted,
                the manifest-specified probability column is read from `df`.
            gate_column: Destination column name (default: "gate_pass").
            mode: Gate mode ("inference" or "training").
            inplace: When True, mutate `df`; otherwise return a copy.
            update_metrics: Forward to `apply_manifest_gates` so Prometheus
                gauges may be updated during live inference.
        """
        artifacts = self.ensure_loaded(label)
        target = df if inplace else df.copy()

        mask, _ = apply_manifest_gates(
            target,
            artifacts,
            prob_series=prob_series,
            mode=mode,
            update_metrics=update_metrics,
        )
        target[gate_column] = mask.astype(bool)
        return target

    def build_decision_payload(
        self,
        label: str,
        df: pd.DataFrame,
        *,
        gate_column: str = "gate_pass",
        include_features: bool = False,
    ) -> Dict[str, object]:
        """
        Serialise a decision payload consumed by trading components.

        The expected workflow is:
            1. Produce probabilities for the desired rows.
            2. Call `annotate_with_gate_pass` to append the boolean gate verdict.
            3. Pass the resulting frame to `build_decision_payload`.

        Args:
            label: Model identifier.
            df: Data frame that *already* includes a probability column and
                the gate decision column (`gate_column`).
            gate_column: Column containing the gate decision (default: "gate_pass").
            include_features: When True, include the full row payload under
                `features`; otherwise only probability/gate/timestamp metadata
                are emitted.
        """
        artifacts = self.ensure_loaded(label)
        prob_col = artifacts.prob_column
        if gate_column not in df.columns:
            raise KeyError(
                f"Data frame missing gate column '{gate_column}' required for payload construction"
            )
        if prob_col not in df.columns:
            raise KeyError(
                f"Data frame missing probability column '{prob_col}' declared by manifest"
            )

        records = df.to_dict(orient="records")
        items: List[Dict[str, object]] = []
        for row in records:
            item: Dict[str, object] = {
                "timestamp": row.get("timestamp"),
                "probability": row.get(prob_col),
                "gate_pass": bool(row.get(gate_column)),
            }
            for key in ("price", "close", "mid_price", "last_price", "bid", "ask"):
                value = row.get(key)
                if value is None or pd.isna(value):
                    continue
                item[key] = value
            if include_features:
                item["features"] = row
            items.append(item)

        return {
            "model": label,
            "artifact_dir": str(artifacts.base_dir),
            "prob_column": prob_col,
            "gate_column": gate_column,
            "items": items,
        }

    def prepare_decision_payload(
        self,
        label: str,
        df: pd.DataFrame,
        *,
        prob_series: Optional[pd.Series] = None,
        gate_column: str = "gate_pass",
        include_features: bool = False,
        update_metrics: bool = True,
    ) -> Dict[str, object]:
        """
        Convenience helper that runs the full pipeline (annotate + serialise).
        """
        annotated = self.annotate_with_gate_pass(
            label,
            df,
            prob_series=prob_series,
            gate_column=gate_column,
            inplace=False,
            update_metrics=update_metrics,
        )
        return self.build_decision_payload(
            label,
            annotated,
            gate_column=gate_column,
            include_features=include_features,
        )


_REGISTRY_SINGLETON: Optional[ManifestRegistry] = None


def get_manifest_registry() -> ManifestRegistry:
    global _REGISTRY_SINGLETON
    if _REGISTRY_SINGLETON is None:
        _REGISTRY_SINGLETON = ManifestRegistry()
    return _REGISTRY_SINGLETON


def _set_manifest_registry_for_tests(registry: Optional[ManifestRegistry]) -> None:
    """
    Test helper that overrides the module-level singleton. External code should
    rely on `get_manifest_registry()`; this function is intentionally private.
    """
    global _REGISTRY_SINGLETON
    _REGISTRY_SINGLETON = registry


# ---------------------------------------------------------------------- #
# Module-level convenience wrappers
# ---------------------------------------------------------------------- #
def annotate_with_gate_pass(
    label: str,
    df: pd.DataFrame,
    *,
    prob_series: Optional[pd.Series] = None,
    gate_column: str = "gate_pass",
    inplace: bool = False,
    update_metrics: bool = True,
) -> pd.DataFrame:
    registry = get_manifest_registry()
    return registry.annotate_with_gate_pass(
        label,
        df,
        prob_series=prob_series,
        gate_column=gate_column,
        inplace=inplace,
        update_metrics=update_metrics,
    )


def build_decision_payload(
    label: str,
    df: pd.DataFrame,
    *,
    gate_column: str = "gate_pass",
    include_features: bool = False,
) -> Dict[str, object]:
    registry = get_manifest_registry()
    return registry.build_decision_payload(
        label,
        df,
        gate_column=gate_column,
        include_features=include_features,
    )


def prepare_decision_payload(
    label: str,
    df: pd.DataFrame,
    *,
    prob_series: Optional[pd.Series] = None,
    gate_column: str = "gate_pass",
    include_features: bool = False,
    update_metrics: bool = True,
) -> Dict[str, object]:
    registry = get_manifest_registry()
    return registry.prepare_decision_payload(
        label,
        df,
        prob_series=prob_series,
        gate_column=gate_column,
        include_features=include_features,
        update_metrics=update_metrics,
    )
