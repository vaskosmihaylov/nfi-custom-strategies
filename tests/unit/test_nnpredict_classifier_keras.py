import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_KERAS_PATH = (
  REPO_ROOT
  / "user_data"
  / "strategies"
  / "**NNPredict"
  / "utils"
  / "ClassifierKeras.py"
)


def _load_classifier_keras(monkeypatch, saved_calls):
  class FakeConfigProto:
    def __init__(self, *args, **kwargs):
      self.gpu_options = types.SimpleNamespace(
        allow_growth=False,
        per_process_gpu_memory_fraction=0.0,
      )

  fake_tf = types.SimpleNamespace(
    compat=types.SimpleNamespace(
      v1=types.SimpleNamespace(
        logging=types.SimpleNamespace(
          ERROR="ERROR",
          set_verbosity=lambda *args, **kwargs: None,
        ),
        ConfigProto=FakeConfigProto,
        Session=lambda *args, **kwargs: object(),
        keras=types.SimpleNamespace(backend=types.SimpleNamespace()),
      )
    ),
    config=types.SimpleNamespace(set_visible_devices=lambda *args, **kwargs: None),
    random=types.SimpleNamespace(set_seed=lambda *args, **kwargs: None),
  )

  def register_keras_serializable(**kwargs):
    def decorator(cls):
      return cls

    return decorator

  def save_model(model, filepath, save_format):
    saved_calls.append((model, filepath, save_format))

  fake_keras = types.SimpleNamespace(
    saving=types.SimpleNamespace(register_keras_serializable=register_keras_serializable),
    models=types.SimpleNamespace(
      save_model=save_model,
      load_model=lambda *args, **kwargs: object(),
    ),
    optimizers=types.SimpleNamespace(Adam=lambda *args, **kwargs: object()),
    callbacks=types.SimpleNamespace(
      EarlyStopping=lambda *args, **kwargs: object(),
      ReduceLROnPlateau=lambda *args, **kwargs: object(),
      ModelCheckpoint=lambda *args, **kwargs: object(),
    ),
    layers=types.SimpleNamespace(Dense=lambda *args, **kwargs: object()),
    Sequential=lambda *args, **kwargs: object(),
  )

  fake_dataframe_utils_module = types.ModuleType("DataframeUtils")

  class FakeDataframeUtils:
    pass

  fake_dataframe_utils_module.DataframeUtils = FakeDataframeUtils

  monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
  monkeypatch.setitem(sys.modules, "keras", fake_keras)
  monkeypatch.setitem(sys.modules, "DataframeUtils", fake_dataframe_utils_module)
  monkeypatch.delitem(sys.modules, "nnpredict_classifier_keras_under_test", raising=False)

  spec = importlib.util.spec_from_file_location(
    "nnpredict_classifier_keras_under_test",
    CLASSIFIER_KERAS_PATH,
  )
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def test_save_clears_new_model_flag(tmp_path, monkeypatch):
  saved_calls = []
  classifier_module = _load_classifier_keras(monkeypatch, saved_calls)

  classifier = classifier_module.ClassifierKeras("BTC/USDT", 12, 8)
  classifier.model = object()
  classifier_module.ClassifierKeras.new_model = True

  model_path = tmp_path / "models" / "NNPredict.keras"
  classifier.save(str(model_path))

  assert saved_calls == [
    (classifier.model, str(model_path), classifier.model_ext),
  ]
  assert classifier_module.ClassifierKeras.new_model is False
