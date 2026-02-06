"""Unit tests for local few-shot models."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import sys

# Ensure `few_shot_learning` package is importable from repo root test runs.
PLUGIN_PARENT = Path(__file__).resolve().parents[2]
if str(PLUGIN_PARENT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_PARENT))

from few_shot_learning.models import (
    SUPPORTED_MODEL_NAMES,
    get_model,
)


def _make_binary_data(n_pos: int = 20, n_neg: int = 20, dim: int = 16, seed: int = 42):
    rng = np.random.default_rng(seed)
    pos = rng.normal(loc=1.0, scale=0.4, size=(n_pos, dim)).astype(np.float32)
    neg = rng.normal(loc=-1.0, scale=0.4, size=(n_neg, dim)).astype(np.float32)
    embeddings = np.vstack([pos, neg])
    labels = np.array([1] * n_pos + [0] * n_neg, dtype=int)
    return embeddings, labels


def test_get_model_supported_names():
    for name in SUPPORTED_MODEL_NAMES:
        model = get_model(name)
        assert model is not None


def test_get_model_unknown_name():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("NotAModel")


def test_linear_svm_fit_predict():
    embeddings, labels = _make_binary_data()
    model = get_model("LinearSVMModel", {"C": 0.5, "max_iter": 500})
    model.fit_step([{"embeddings": embeddings, "labels": labels}])

    out = model.predict({"embeddings": embeddings[:8]})
    assert "scores" in out
    assert "probs" in out
    assert out["scores"].shape == (8,)
    assert out["probs"].shape == (8,)
    assert np.all((out["probs"] >= 0.0) & (out["probs"] <= 1.0))


def test_rocchio_fit_predict():
    embeddings, labels = _make_binary_data()
    model = get_model(
        "RocchioPrototypeModel",
        {"mode": "proto_softmax", "beta": 1.0, "gamma": 1.0, "temperature": 1.0},
    )
    model.fit_step([{"embeddings": embeddings, "labels": labels}])

    out = model.predict({"embeddings": embeddings[:8]})
    assert "scores" in out
    assert out["scores"].shape == (8,)
    assert np.all((out["scores"] >= 0.0) & (out["scores"] <= 1.0))


def test_nca_fit_predict():
    embeddings, labels = _make_binary_data(dim=20)
    model = get_model("NCAMetricLearningModel", {"n_components": 8, "max_iter": 10})
    model.fit_step([{"embeddings": embeddings, "labels": labels}])

    out = model.predict({"embeddings": embeddings[:10]})
    assert "scores" in out
    assert "embeddings_transformed" in out
    assert out["scores"].shape == (10,)
    assert out["embeddings_transformed"].shape == (10, 8)


def test_lmnn_fit_predict():
    embeddings, labels = _make_binary_data(dim=20)
    model = get_model(
        "LMNNMetricLearningModel",
        {"n_components": 8, "k": 2, "max_iter": 5, "learning_rate": 0.01},
    )
    model.fit_step([{"embeddings": embeddings, "labels": labels}])

    out = model.predict({"embeddings": embeddings[:10]})
    assert "scores" in out
    assert "embeddings_transformed" in out
    assert out["scores"].shape == (10,)
    assert out["embeddings_transformed"].shape == (10, 8)


def test_graph_label_propagation_transductive():
    embeddings, labels = _make_binary_data(n_pos=30, n_neg=30, dim=8)

    y_lp = np.full(len(labels), fill_value=-1, dtype=int)
    pos_idx = np.where(labels == 1)[0][:5]
    neg_idx = np.where(labels == 0)[0][:5]
    y_lp[pos_idx] = 1
    y_lp[neg_idx] = 0

    ids = [f"id_{idx}" for idx in range(len(labels))]

    model = get_model(
        "GraphLabelPropagationModel",
        {"n_neighbors": 10, "alpha": 0.2, "max_iter": 30},
    )
    model.fit_step([{"embeddings": embeddings, "labels": y_lp, "ids": ids}])

    out = model.predict({"embeddings": embeddings, "ids": ids})
    assert "p_pos" in out
    assert "uncertainty" in out
    assert "scores" in out
    assert out["p_pos"].shape == (len(labels),)
    assert np.all((out["p_pos"] >= 0.0) & (out["p_pos"] <= 1.0))
    assert np.all((out["uncertainty"] >= 0.0) & (out["uncertainty"] <= 1.0))


def test_graph_requires_labeled_points():
    embeddings, labels = _make_binary_data(n_pos=10, n_neg=10, dim=8)
    y_lp = np.full(len(labels), fill_value=-1, dtype=int)

    model = get_model("GraphLabelPropagationModel", {"n_neighbors": 5})
    with pytest.raises(ValueError, match="at least one labeled sample"):
        model.fit_step([{"embeddings": embeddings, "labels": y_lp}])


def test_rocchio_rejects_alpha_hyperparam():
    with pytest.raises(ValueError, match="does not support 'alpha'"):
        get_model("RocchioPrototypeModel", {"alpha": 0.5})


def test_graph_unseen_ids_use_neutral_probability():
    embeddings, labels = _make_binary_data(n_pos=20, n_neg=20, dim=8)
    y_lp = np.full(len(labels), fill_value=-1, dtype=int)
    y_lp[:4] = labels[:4]
    y_lp[20:24] = labels[20:24]

    fit_ids = [f"fit_{i}" for i in range(len(labels))]
    predict_ids = [f"pred_{i}" for i in range(len(labels))]

    model = get_model("GraphLabelPropagationModel", {"n_neighbors": 8})
    model.fit_step([{"embeddings": embeddings, "labels": y_lp, "ids": fit_ids}])
    out = model.predict({"embeddings": embeddings, "ids": predict_ids})

    assert np.allclose(out["p_pos"], 0.5)
    assert np.allclose(out["uncertainty"], 1.0)


def test_stack_embeddings_labels_rejects_mixed_batches():
    from few_shot_learning.models import _stack_embeddings_and_labels

    emb = np.random.randn(4, 8).astype(np.float32)
    with pytest.raises(ValueError, match="Mixed batches"):
        _stack_embeddings_and_labels(
            [{"embeddings": emb, "labels": np.array([0, 1, 0, 1])}, {"embeddings": emb}],
            require_labels=False,
        )
