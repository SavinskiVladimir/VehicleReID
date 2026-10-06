"""Тесты scripts/benchmark_models.py на крошечной настоящей ONNX-модели (фабрика из conftest.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_models as bm  # noqa: E402


def test_time_session_measures_a_real_model_with_and_without_thread_limit(build_toy_embedder, tmp_path):
    model = build_toy_embedder(tmp_path / "m.onnx", out_dim=8, size=32)
    for threads in (0, 1):
        median, p95 = bm.time_session(str(model), threads, runs=5, warmup=1)
        assert 0 < median <= p95


def test_run_prints_a_row_per_thread_setting_and_skips_missing_models(build_toy_embedder, tmp_path, capsys):
    model = build_toy_embedder(tmp_path / "m.onnx", out_dim=8, size=32)
    args = bm.build_parser().parse_args(
        ["--detector", str(tmp_path / "нет.onnx"), "--reid", str(model), "--runs", "3", "--warmup", "1", "--threads", "0", "2"])

    assert bm.run(args) == 0

    out = capsys.readouterr().out
    assert "файл не найден" in out and "пропуск" in out          # детектора нет - не падаем
    assert out.count("медиана") == 2 and "по умолчанию" in out    # два режима потоков для ReID


def test_every_onnxruntime_optimization_level_works(build_toy_embedder, tmp_path):
    model = build_toy_embedder(tmp_path / "m.onnx", out_dim=8, size=32)
    for level in ("all", "extended", "basic"):
        median, _ = bm.time_session(str(model), 1, runs=2, warmup=1, opt_level=level)
        assert median > 0


def test_pytorch_reference_loads_a_project_checkpoint_and_times_it(tmp_path):
    import pytest
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml" / "reid"))
    from models.efficientnetv2_reid import VehicleReIDNet

    model = VehicleReIDNet(num_classes=3, embedding_dim=16, pretrained=False)
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": model.state_dict(), "num_train_pids": 3, "embedding_dim": 16}, checkpoint)

    for channels_last in (False, True):
        median, p95 = bm.time_torch(str(checkpoint), 2, runs=1, warmup=1, channels_last=channels_last, hw=64)
        assert 0 < median <= p95