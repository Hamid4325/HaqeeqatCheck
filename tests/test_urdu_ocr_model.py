import pytest

torch = pytest.importorskip("torch")


def test_model_forward_shape():
    from ingestion.urdu_ocr.model import Model

    model = Model(num_class=10, device="cpu")
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 32, 400))
    assert tuple(out.shape) == (1, 400, 10)


def test_model_loads_state_dict_keys_match():
    from ingestion.urdu_ocr.model import Model

    model = Model(num_class=541, device="cpu")
    keys = set(model.state_dict().keys())
    assert "FeatureExtraction.ConvNet.inc.double_conv.0.weight" in keys
    assert "SequenceModeling.0.rnn.weight_ih_l0" in keys
    assert "Prediction.weight" in keys
