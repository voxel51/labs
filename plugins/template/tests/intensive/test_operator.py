import pytest
import fiftyone.operators as foo
import fiftyone.zoo as foz

@pytest.fixture
def dataset():
    return foz.load_zoo_dataset("quickstart").limit(10)

@pytest.mark.unit
def test_plugin_basic_functionality(dataset):
    ctx = {
        "dataset": dataset._dataset,
        "view": dataset,
        "params": {
            "message": "Hello from template plugin",
        },
        "delegated": False,
    }
    result = foo.execute_operator(
        "@51labs/template/template_operator", ctx
    )
    assert result.result["status"] == "success"  # type: ignore
    assert "Hello" in result.result["message"]  # type: ignore
    assert result.result["sample_count"] == 10  # type: ignore
