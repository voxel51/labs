"""Self-contained few-shot models used by the panel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.neighbors import NeighborhoodComponentsAnalysis, NearestNeighbors
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import LinearSVC


SUPPORTED_MODEL_NAMES = [
    "LinearSVMModel",
    "RocchioPrototypeModel",
    "NCAMetricLearningModel",
    "LMNNMetricLearningModel",
    "GraphLabelPropagationModel",
]

BatchDict = dict[str, Any]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute a numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _as_float32_2d_embeddings(embeddings: Any) -> np.ndarray:
    """Convert input embeddings to a `(N, D)` float32 NumPy array."""
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected embeddings with shape (N, D), got {arr.shape}")
    return arr


def _normalize_binary_labels(labels: Any, model_name: str) -> np.ndarray:
    """Normalize binary labels to `{0, 1}` where `1` denotes positives."""
    y = np.asarray(labels).astype(int).reshape(-1)
    uniq = set(np.unique(y).tolist())

    if uniq.issubset({0, 1}):
        return y
    if uniq.issubset({-1, 1}):
        return (y == 1).astype(int)

    raise ValueError(
        f"{model_name} expects binary labels in {{0,1}} or {{-1,+1}}, got {sorted(uniq)}"
    )


def _stack_embeddings_and_labels(
    batches: Sequence[BatchDict], *, require_labels: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Concatenate embeddings and labels across a sequence of batches."""
    if not batches:
        raise ValueError("Expected at least one batch")

    embeddings_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    labels_missing = False

    for batch in batches:
        if "embeddings" not in batch:
            raise KeyError("Batch is missing 'embeddings' key.")

        embeddings = _as_float32_2d_embeddings(batch["embeddings"])
        embeddings_list.append(embeddings)

        labels = batch.get("labels", None)
        if labels is None:
            labels_missing = True
            continue

        labels_arr = np.asarray(labels).reshape(-1)
        if len(labels_arr) != len(embeddings):
            raise ValueError(
                "Batch labels length must match embeddings rows: "
                f"{len(labels_arr)} != {len(embeddings)}"
            )
        labels_list.append(labels_arr)

    stacked_embeddings = np.vstack(embeddings_list)

    if labels_missing:
        if require_labels:
            raise ValueError("Labels are required for this model")
        if labels_list:
            raise ValueError(
                "Mixed batches with and without labels are not supported. "
                "Provide labels for all batches or none."
            )
        return stacked_embeddings, None

    if not labels_list:
        if require_labels:
            raise ValueError("Labels are required for this model")
        return stacked_embeddings, None

    stacked_labels = np.concatenate(labels_list)
    if len(stacked_labels) != len(stacked_embeddings):
        raise ValueError(
            "Concatenated labels length must match concatenated embeddings rows: "
            f"{len(stacked_labels)} != {len(stacked_embeddings)}"
        )
    return stacked_embeddings, stacked_labels


def _normalize_ids(ids: Any, expected_len: int) -> list[str] | None:
    """Normalize IDs to strings and validate length against embeddings rows."""
    if ids is None:
        return None

    arr = np.asarray(ids, dtype=object).reshape(-1)
    if len(arr) != expected_len:
        raise ValueError(f"'ids' length must match embeddings rows: {len(arr)} != {expected_len}")

    normalized: list[str] = []
    for value in arr.tolist():
        if isinstance(value, np.generic):
            value = value.item()
        normalized.append(str(value))
    return normalized


class _BaseModel:
    """Minimal model interface used by the panel training loop."""

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        """Store hyperparameters and initialize compatibility flags."""
        self.hyperparams = dict(hyperparams or {})
        self.enable_postprocess = False

    def build_output_processor(self, classes: list[str] | None = None) -> None:
        """Compatibility no-op for callers expecting this method."""
        # Panel calls this opportunistically; keep it as a compatibility no-op.
        self._output_classes = list(classes or [])

    def predict(self, batch: BatchDict) -> BatchDict:
        """Run model inference on a single batch dictionary."""
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch)!r}")
        return self._predict_dict(dict(batch))

    def _predict_dict(self, batch: BatchDict) -> BatchDict:
        """Internal predict implementation overridden by subclasses."""
        raise NotImplementedError


class LinearSVMModel(_BaseModel):
    """Binary classifier based on `sklearn.svm.LinearSVC`."""

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        """Initialize the SVM estimator and model state."""
        super().__init__(hyperparams)
        self._fitted = False

        C = float(self.hyperparams.get("C", 1.0))
        max_iter = int(self.hyperparams.get("max_iter", 10000))
        random_state = int(self.hyperparams.get("random_state", 42))
        self._estimator = LinearSVC(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            dual="auto",
        )

    def fit_step(self, batches: Sequence[BatchDict]) -> None:
        """Fit the linear SVM on labeled embedding batches."""
        embeddings, labels = _stack_embeddings_and_labels(batches, require_labels=True)
        labels = _normalize_binary_labels(labels, "LinearSVMModel")

        if len(np.unique(labels)) < 2:
            raise ValueError(
                "LinearSVMModel requires at least 2 classes. "
                "Ensure you have both positive and negative examples."
            )

        self._estimator.fit(embeddings, labels)
        self._fitted = True

    def _predict_dict(self, batch: BatchDict) -> BatchDict:
        """Predict SVM scores and sigmoid probabilities."""
        if "embeddings" not in batch:
            raise KeyError("Batch is missing 'embeddings' key.")

        embeddings = _as_float32_2d_embeddings(batch["embeddings"])
        if self._fitted:
            scores = np.asarray(self._estimator.decision_function(embeddings), dtype=np.float32)
            probs = _sigmoid(scores).astype(np.float32)
        else:
            scores = np.zeros(len(embeddings), dtype=np.float32)
            probs = np.full(len(embeddings), 0.5, dtype=np.float32)

        batch["scores"] = scores
        batch["probs"] = probs
        return batch

    @property
    def is_fitted(self) -> bool:
        """Whether `fit_step` has completed successfully."""
        return self._fitted


class RocchioPrototypeModel(_BaseModel):
    """Centroid-based binary scorer with Rocchio/prototype variants."""

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        """Initialize Rocchio scoring configuration and state."""
        super().__init__(hyperparams)
        self._fitted = False

        if "alpha" in self.hyperparams:
            raise ValueError(
                "RocchioPrototypeModel does not support 'alpha' in this plugin port."
            )

        self._mode = str(self.hyperparams.get("mode", "proto_softmax"))
        self._beta = float(self.hyperparams.get("beta", 1.0))
        self._gamma = float(self.hyperparams.get("gamma", 1.0))
        self._temperature = float(self.hyperparams.get("temperature", 1.0))
        self._normalize_embeddings = bool(self.hyperparams.get("normalize_embeddings", False))

        self._pos_centroid: np.ndarray | None = None
        self._neg_centroid: np.ndarray | None = None
        self._query_vector: np.ndarray | None = None

    def _transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply optional L2 normalization to embeddings."""
        if not self._normalize_embeddings:
            return embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return embeddings / norms

    def fit_step(self, batches: Sequence[BatchDict]) -> None:
        """Fit class centroids from labeled embedding batches."""
        embeddings, labels = _stack_embeddings_and_labels(batches, require_labels=True)
        labels = _normalize_binary_labels(labels, "RocchioPrototypeModel")

        if len(np.unique(labels)) < 2:
            raise ValueError(
                "RocchioPrototypeModel requires at least 2 classes. "
                "Ensure you have both positive and negative examples."
            )

        transformed = self._transform(embeddings)
        pos = transformed[labels == 1]
        neg = transformed[labels == 0]

        self._pos_centroid = pos.mean(axis=0)
        self._neg_centroid = neg.mean(axis=0)
        self._query_vector = self._beta * self._pos_centroid - self._gamma * self._neg_centroid
        self._fitted = True

    def _predict_dict(self, batch: BatchDict) -> BatchDict:
        """Predict Rocchio/prototype confidence scores in `[0, 1]`."""
        if "embeddings" not in batch:
            raise KeyError("Batch is missing 'embeddings' key.")

        embeddings = _as_float32_2d_embeddings(batch["embeddings"])
        transformed = self._transform(embeddings)
        batch["embeddings_transformed"] = transformed

        if not self._fitted:
            batch["scores"] = np.zeros(len(embeddings), dtype=np.float32)
            return batch

        temp = max(self._temperature, 1e-8)
        if self._mode == "proto_softmax":
            dist_to_pos = np.linalg.norm(transformed - self._pos_centroid, axis=1)
            dist_to_neg = np.linalg.norm(transformed - self._neg_centroid, axis=1)
            scores = _sigmoid((dist_to_neg - dist_to_pos) / temp)
        elif self._mode == "rocchio_sigmoid":
            raw = transformed @ self._query_vector
            scores = _sigmoid(raw / temp)
        else:
            raise ValueError(f"Unknown Rocchio mode: {self._mode}")

        batch["scores"] = np.asarray(scores, dtype=np.float32)
        return batch

    @property
    def is_fitted(self) -> bool:
        """Whether `fit_step` has completed successfully."""
        return self._fitted


class NCAMetricLearningModel(_BaseModel):
    """Metric learning model backed by sklearn NCA."""

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        """Initialize NCA model state."""
        super().__init__(hyperparams)
        self._fitted = False
        self._estimator: NeighborhoodComponentsAnalysis | None = None
        self._pos_centroid: np.ndarray | None = None
        self._neg_centroid: np.ndarray | None = None

    def fit_step(self, batches: Sequence[BatchDict]) -> None:
        """Fit NCA transform and positive/negative centroids."""
        embeddings, labels = _stack_embeddings_and_labels(batches, require_labels=True)
        labels = _normalize_binary_labels(labels, "NCAMetricLearningModel")

        if len(np.unique(labels)) < 2:
            raise ValueError(
                "NCAMetricLearningModel requires at least 2 classes. "
                "Ensure you have both positive and negative examples."
            )

        n_components = int(self.hyperparams.get("n_components", 64))
        n_components = max(1, min(n_components, embeddings.shape[1]))
        max_iter = int(self.hyperparams.get("max_iter", 100))
        random_state = int(self.hyperparams.get("random_state", 42))

        self._estimator = NeighborhoodComponentsAnalysis(
            n_components=n_components,
            max_iter=max_iter,
            random_state=random_state,
        )
        self._estimator.fit(embeddings, labels)

        transformed = self._estimator.transform(embeddings)
        self._pos_centroid = transformed[labels == 1].mean(axis=0)
        self._neg_centroid = transformed[labels == 0].mean(axis=0)
        self._fitted = True

    def _predict_dict(self, batch: BatchDict) -> BatchDict:
        """Project embeddings and score by centroid distance difference."""
        if "embeddings" not in batch:
            raise KeyError("Batch is missing 'embeddings' key.")

        embeddings = _as_float32_2d_embeddings(batch["embeddings"])
        if self._fitted:
            transformed = self._estimator.transform(embeddings)
            dist_to_pos = np.linalg.norm(transformed - self._pos_centroid, axis=1)
            dist_to_neg = np.linalg.norm(transformed - self._neg_centroid, axis=1)
            scores = dist_to_neg - dist_to_pos
        else:
            transformed = embeddings
            scores = np.zeros(len(embeddings), dtype=np.float32)

        batch["embeddings_transformed"] = np.asarray(transformed, dtype=np.float32)
        batch["scores"] = np.asarray(scores, dtype=np.float32)
        return batch

    @property
    def is_fitted(self) -> bool:
        """Whether `fit_step` has completed successfully."""
        return self._fitted


def _as_int_binary_labels(labels: np.ndarray) -> np.ndarray:
    """Normalize LMNN labels to `{0, 1}`."""
    return _normalize_binary_labels(labels, "LMNNMetricLearningModel")


@dataclass
class _LMNNHyperparams:
    n_components: int = 64
    k: int = 3
    n_impostors: int = 10
    margin: float = 1.0
    mu: float = 0.5
    reg: float = 1e-4
    max_iter: int = 200
    learning_rate: float = 1e-2
    random_state: int = 42


class LMNNLinearTransformer:
    """Lightweight LMNN-style linear transformer for small few-shot support sets."""

    def __init__(self, hyperparams: _LMNNHyperparams) -> None:
        """Store LMNN hyperparameters and initialize learned components."""
        self._hp = hyperparams
        self.components_: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> "LMNNLinearTransformer":
        """Fit a linear transform with an LMNN-inspired objective."""
        X = np.asarray(embeddings, dtype=np.float32)
        y = _as_int_binary_labels(labels)

        n, d = X.shape
        if n < 2:
            raise ValueError("LMNNMetricLearningModel requires at least 2 samples.")
        if len(np.unique(y)) < 2:
            raise ValueError(
                "LMNNMetricLearningModel requires at least 2 classes. "
                "Ensure you have both positive and negative examples."
            )

        k = max(1, int(self._hp.k))
        n_impostors = max(1, int(self._hp.n_impostors))
        out_dim = min(max(1, int(self._hp.n_components)), d)

        rng = np.random.default_rng(int(self._hp.random_state))
        torch.manual_seed(int(self._hp.random_state))

        target_pairs: list[tuple[int, int]] = []
        for cls in np.unique(y).tolist():
            idxs = np.where(y == cls)[0]
            if len(idxs) <= 1:
                continue

            X_cls = X[idxs]
            nn = NearestNeighbors(n_neighbors=min(k + 1, len(idxs)), metric="euclidean")
            nn.fit(X_cls)
            neigh = nn.kneighbors(X_cls, return_distance=False)
            for row, i_global in enumerate(idxs):
                for j_local in neigh[row, 1:]:
                    j_global = int(idxs[int(j_local)])
                    if i_global != j_global:
                        target_pairs.append((int(i_global), int(j_global)))

        if not target_pairs:
            self.components_ = np.eye(d, out_dim, dtype=np.float32)
            return self

        impostors_for_i: dict[int, np.ndarray] = {}
        for i in range(n):
            opposite = np.where(y != y[i])[0]
            if len(opposite) == 0:
                impostors_for_i[i] = np.array([], dtype=int)
                continue

            X_opp = X[opposite]
            nn = NearestNeighbors(
                n_neighbors=min(n_impostors, len(opposite)),
                metric="euclidean",
            )
            nn.fit(X_opp)
            neigh = nn.kneighbors(X[i : i + 1], return_distance=False)[0]
            impostors_for_i[i] = opposite[neigh].astype(int)

        X_t = torch.from_numpy(X)

        init_L = np.zeros((d, out_dim), dtype=np.float32)
        init_L[:out_dim, :out_dim] = np.eye(out_dim, dtype=np.float32)
        init_L += 0.01 * rng.standard_normal(size=init_L.shape).astype(np.float32)
        L = torch.nn.Parameter(torch.from_numpy(init_L))

        opt = torch.optim.Adam([L], lr=float(self._hp.learning_rate))

        pair_i = torch.tensor([p[0] for p in target_pairs], dtype=torch.long)
        pair_j = torch.tensor([p[1] for p in target_pairs], dtype=torch.long)

        margin = float(self._hp.margin)
        mu = float(self._hp.mu)
        reg = float(self._hp.reg)

        for _ in range(int(self._hp.max_iter)):
            opt.zero_grad(set_to_none=True)

            xi = X_t[pair_i]
            xj = X_t[pair_j]
            diff_ij = xi - xj
            proj_ij = diff_ij @ L
            d_ij = (proj_ij * proj_ij).sum(dim=1)
            loss_pull = d_ij.mean()

            hinge_terms: list[torch.Tensor] = []
            for p in range(pair_i.numel()):
                i = int(pair_i[p].item())
                impostors = impostors_for_i.get(i)
                if impostors is None or len(impostors) == 0:
                    continue

                xl = X_t[torch.from_numpy(impostors)]
                di = X_t[i : i + 1]
                diff_il = di - xl
                proj_il = diff_il @ L
                d_il = (proj_il * proj_il).sum(dim=1)
                hinge = torch.relu(margin + d_ij[p] - d_il)
                hinge_terms.append(hinge.mean())

            if hinge_terms:
                loss_push = torch.stack(hinge_terms).mean()
            else:
                loss_push = torch.zeros((), dtype=torch.float32)

            loss = (1.0 - mu) * loss_pull + mu * loss_push + reg * (L * L).mean()
            loss.backward()
            opt.step()

        self.components_ = L.detach().cpu().numpy().astype(np.float32)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply the learned linear transformation to embeddings."""
        if self.components_ is None:
            raise ValueError("LMNNLinearTransformer is not fitted yet")
        X = np.asarray(embeddings, dtype=np.float32)
        return X @ self.components_


class LMNNMetricLearningModel(_BaseModel):
    """Few-shot metric learning model using a lightweight LMNN transformer."""

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        """Initialize LMNN model state."""
        super().__init__(hyperparams)
        self._fitted = False
        self._transformer: LMNNLinearTransformer | None = None
        self._pos_centroid: np.ndarray | None = None
        self._neg_centroid: np.ndarray | None = None

    def _get_hyperparams(self) -> _LMNNHyperparams:
        """Resolve hyperparameters with defaults."""
        return _LMNNHyperparams(
            n_components=int(self.hyperparams.get("n_components", 64)),
            k=int(self.hyperparams.get("k", 3)),
            n_impostors=int(self.hyperparams.get("n_impostors", 10)),
            margin=float(self.hyperparams.get("margin", 1.0)),
            mu=float(self.hyperparams.get("mu", 0.5)),
            reg=float(self.hyperparams.get("reg", 1e-4)),
            max_iter=int(self.hyperparams.get("max_iter", 200)),
            learning_rate=float(self.hyperparams.get("learning_rate", 1e-2)),
            random_state=int(self.hyperparams.get("random_state", 42)),
        )

    def fit_step(self, batches: Sequence[BatchDict]) -> None:
        """Fit LMNN transform and class centroids on labeled batches."""
        embeddings, labels = _stack_embeddings_and_labels(batches, require_labels=True)
        labels = _as_int_binary_labels(labels)

        if len(np.unique(labels)) < 2:
            raise ValueError(
                "LMNNMetricLearningModel requires at least 2 classes. "
                "Ensure you have both positive and negative examples."
            )

        hp = self._get_hyperparams()
        hp.n_components = min(max(1, hp.n_components), embeddings.shape[1])

        transformer = LMNNLinearTransformer(hp)
        transformer.fit(embeddings, labels)

        transformed = transformer.transform(embeddings)
        self._pos_centroid = transformed[labels == 1].mean(axis=0)
        self._neg_centroid = transformed[labels == 0].mean(axis=0)

        self._transformer = transformer
        self._fitted = True

    def _predict_dict(self, batch: BatchDict) -> BatchDict:
        """Predict LMNN centroid-distance scores for a batch."""
        if "embeddings" not in batch:
            raise KeyError("Batch is missing 'embeddings' key.")

        embeddings = _as_float32_2d_embeddings(batch["embeddings"])
        if self._fitted:
            transformed = self._transformer.transform(embeddings)
            dist_to_pos = np.linalg.norm(transformed - self._pos_centroid, axis=1)
            dist_to_neg = np.linalg.norm(transformed - self._neg_centroid, axis=1)
            scores = dist_to_neg - dist_to_pos
        else:
            transformed = embeddings
            scores = np.zeros(len(embeddings), dtype=np.float32)

        batch["embeddings_transformed"] = np.asarray(transformed, dtype=np.float32)
        batch["scores"] = np.asarray(scores, dtype=np.float32)
        return batch

    @property
    def is_fitted(self) -> bool:
        """Whether `fit_step` has completed successfully."""
        return self._fitted


def _uncertainty_from_p_pos(p_pos: np.ndarray) -> np.ndarray:
    """Compute normalized binary entropy from positive-class probabilities."""
    p = np.asarray(p_pos, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.5)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return (entropy / np.log(2.0)).astype(np.float32)


class GraphLabelPropagationModel(_BaseModel):
    """Transductive graph label propagation model used by the panel."""

    supports_transductive: bool = True

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        """Initialize transductive graph-propagation configuration and state."""
        super().__init__(hyperparams)
        self._fitted = False
        self._estimator: LabelSpreading | None = None

        self._id_to_p_pos: dict[str, float] = {}
        self._id_to_uncertainty: dict[str, float] = {}

        self._n_neighbors = int(self.hyperparams.get("n_neighbors", 30))
        self._alpha = float(self.hyperparams.get("alpha", 0.2))
        self._max_iter = int(self.hyperparams.get("max_iter", 30))
        self._tol = float(self.hyperparams.get("tol", 1e-3))

    def fit_step(self, batches: Sequence[BatchDict]) -> None:
        """Fit label propagation on support plus unlabeled candidate nodes."""
        embeddings, labels = _stack_embeddings_and_labels(batches, require_labels=True)
        y = np.asarray(labels).astype(int).reshape(-1)
        if len(y) != len(embeddings):
            raise ValueError(
                "GraphLabelPropagationModel labels length must match embeddings rows"
            )

        unique_vals = set(np.unique(y).tolist())
        if not unique_vals.issubset({-1, 0, 1}):
            raise ValueError(
                "GraphLabelPropagationModel requires labels in {-1, 0, 1} "
                "(-1 means unlabeled)"
            )

        labeled = y[y != -1]
        if len(labeled) == 0:
            raise ValueError("GraphLabelPropagationModel requires at least one labeled sample")
        if len(np.unique(labeled)) < 2:
            raise ValueError(
                "GraphLabelPropagationModel requires both positive and negative labeled samples"
            )

        ids: list[str] | None = []
        for batch in batches:
            batch_embeddings = _as_float32_2d_embeddings(batch["embeddings"])
            batch_ids = _normalize_ids(batch.get("ids", None), len(batch_embeddings))
            if batch_ids is None:
                ids = None
                break
            ids.extend(batch_ids)
        if ids is not None and len(ids) != len(embeddings):
            raise ValueError("'ids' length must match embeddings rows")

        n_neighbors = max(1, min(self._n_neighbors, max(1, len(embeddings) - 1)))
        estimator = LabelSpreading(
            kernel="knn",
            n_neighbors=n_neighbors,
            alpha=self._alpha,
            max_iter=self._max_iter,
            tol=self._tol,
        )
        estimator.fit(embeddings, y)

        dist = np.asarray(estimator.label_distributions_, dtype=np.float32)
        classes = np.asarray(estimator.classes_, dtype=int)
        if 1 not in classes:
            raise RuntimeError(
                f"LabelSpreading did not produce positive class probabilities: classes={classes.tolist()}"
            )
        pos_col = int(np.where(classes == 1)[0][0])
        p_pos = dist[:, pos_col]
        p_pos = np.nan_to_num(p_pos, nan=0.5)
        p_pos = np.clip(p_pos, 0.0, 1.0).astype(np.float32)
        uncertainty = _uncertainty_from_p_pos(p_pos)

        self._id_to_p_pos = {}
        self._id_to_uncertainty = {}
        if ids is not None:
            for sample_id, prob, uncert in zip(ids, p_pos.tolist(), uncertainty.tolist()):
                self._id_to_p_pos[sample_id] = float(prob)
                self._id_to_uncertainty[sample_id] = float(uncert)

        self._estimator = estimator
        self._fitted = True

    def _predict_dict(self, batch: BatchDict) -> BatchDict:
        """Predict transductive probabilities for IDs seen during fit."""
        if "embeddings" not in batch:
            raise KeyError("Batch is missing 'embeddings' key.")

        embeddings = _as_float32_2d_embeddings(batch["embeddings"])
        n = len(embeddings)
        ids = _normalize_ids(batch.get("ids", None), n)

        if not self._fitted:
            batch["p_pos"] = np.full(n, 0.5, dtype=np.float32)
            batch["uncertainty"] = np.ones(n, dtype=np.float32)
            batch["scores"] = batch["p_pos"]
            return batch

        if ids is not None and self._id_to_p_pos:
            p_pos = np.array(
                [self._id_to_p_pos.get(sample_id, np.nan) for sample_id in ids],
                dtype=np.float32,
            )
            uncertainty = np.array(
                [self._id_to_uncertainty.get(sample_id, np.nan) for sample_id in ids],
                dtype=np.float32,
            )

            missing = np.isnan(p_pos)
            if np.any(missing):
                p_pos[missing] = 0.5
                uncertainty[missing] = 1.0

            batch["p_pos"] = p_pos
            batch["uncertainty"] = uncertainty
            batch["scores"] = p_pos
            return batch

        # This model is transductive in the plugin workflow; unseen IDs get neutral defaults.
        batch["p_pos"] = np.full(n, 0.5, dtype=np.float32)
        batch["uncertainty"] = np.ones(n, dtype=np.float32)
        batch["scores"] = batch["p_pos"]
        return batch

    @property
    def is_fitted(self) -> bool:
        """Whether `fit_step` has completed successfully."""
        return self._fitted


def get_model(name: str, hyperparams: dict[str, Any] | None = None) -> _BaseModel:
    """Instantiate one of the plugin's built-in few-shot model classes."""
    hyperparams = dict(hyperparams or {})

    if name == "LinearSVMModel":
        return LinearSVMModel(hyperparams)
    if name == "RocchioPrototypeModel":
        return RocchioPrototypeModel(hyperparams)
    if name == "NCAMetricLearningModel":
        return NCAMetricLearningModel(hyperparams)
    if name == "LMNNMetricLearningModel":
        return LMNNMetricLearningModel(hyperparams)
    if name == "GraphLabelPropagationModel":
        return GraphLabelPropagationModel(hyperparams)

    raise ValueError(
        f"Unknown model: '{name}'. Available models: {SUPPORTED_MODEL_NAMES}"
    )


__all__ = [
    "SUPPORTED_MODEL_NAMES",
    "get_model",
    "LinearSVMModel",
    "RocchioPrototypeModel",
    "NCAMetricLearningModel",
    "LMNNMetricLearningModel",
    "GraphLabelPropagationModel",
]
