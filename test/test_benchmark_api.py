"""Тесты scripts/benchmark_api.py: расчёт метрик и сам прогон на подставном сервере (httpx.MockTransport)."""

import sys
from pathlib import Path

import httpx
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_api as bench  # noqa: E402


def parse(*argv):
    return bench.build_parser().parse_args([str(a) for a in argv])


class TestMetrics:
    def test_server_timing_is_parsed(self):
        assert bench.parse_server_timing("detect;dur=41.2, embed;dur=88.0, total;dur=140.5") == {
            "detect": 41.2, "embed": 88.0, "total": 140.5}
        assert bench.parse_server_timing("") == {} and bench.parse_server_timing(None) == {}

    def test_percentile_uses_nearest_rank(self):
        values = list(range(1, 21))                       # 1..20
        assert bench.percentile(values, 0.95) == 19       # 19-й из 20
        assert bench.percentile([5], 0.95) == 5

    def test_verdict_uses_p95_for_the_budget(self):
        assert bench.verdict(median=80, p95=95, budget=100) == "выполнено"
        assert bench.verdict(median=80, p95=150, budget=100) == "на грани (p95 выше)"
        assert bench.verdict(median=120, p95=150, budget=100) == "НЕ выполнено"

    @staticmethod
    def _run(detect=40, crop=1, embed=90, search=10, detect_wall=60, search_wall=200, with_detect=True):
        stages = {"crop": crop, "embed": embed, "search": search}
        if with_detect:
            stages["detect"] = detect
        return {"detect_wall": detect_wall, "search_wall": search_wall, "detect_stages": {"detect": detect}, "search_stages": stages}

    def test_crop_metric_is_detect_plus_crop_and_total_is_client_side(self):
        rows, warnings = bench.summarize([self._run()] * 5, "cpu")
        by_key = {r["key"]: r for r in rows}
        assert by_key["crop"]["median"] == 41 and by_key["embed"]["median"] == 90
        assert by_key["total"]["median"] == 260           # 60 + 200, замер клиента
        assert warnings == []

    def test_budgets_follow_the_profile(self):
        cpu = {r["key"]: r["budget"] for r in bench.summarize([self._run()], "cpu")[0]}
        gpu = {r["key"]: r["budget"] for r in bench.summarize([self._run()], "gpu")[0]}
        assert cpu == {"crop": 100, "embed": 300, "search": 30, "total": 700}
        assert gpu == {"crop": 3, "embed": 100, "search": 30, "total": 300}

    def test_missing_detect_stage_is_reported(self):
        # Так выглядел второй скриншот: в /api/search нет этапа detect.
        _, warnings = bench.summarize([self._run(with_detect=False)], "cpu")
        assert warnings and "detect" in warnings[0]


def _server(detections, search_timing="detect;dur=40.0, crop;dur=1.0, embed;dur=90.0, search;dur=10.0, total;dur=150.0"):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/observations":
            return httpx.Response(200, json={"items": [], "total": 11580, "limit": 1, "offset": 0})
        if request.url.path == "/api/detect":
            return httpx.Response(200, headers={"Server-Timing": "read;dur=2.0, save;dur=20.0, detect;dur=40.0, total;dur=63.0"},
                                  json={"upload_id": "u1", "detections": detections})
        return httpx.Response(200, headers={"Server-Timing": search_timing}, json={"results": []})
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")


@pytest.fixture
def frames(tmp_path):
    for i in range(3):
        Image.new("RGB", (64, 48), (i * 50, 90, 90)).save(tmp_path / f"f{i}.png")
    return tmp_path


class TestRun:
    DET = [{"index": 0, "bbox": {"x1": 0, "y1": 0, "x2": 9, "y2": 9}, "confidence": 0.9}]

    def test_prints_a_table_against_the_budgets(self, frames, capsys):
        code = bench.run(parse(frames, "-n", "6", "--warmup", "2"), client=_server(self.DET))

        out = capsys.readouterr().out
        assert code == 0
        assert "Замеров: 6" in out and "11580" in out
        assert "Кроп ТС" in out and "НЕ выполнено" not in out and out.count("выполнено") >= 4   # всё в бюджете
        assert "требование ТЗ - до 100 000" in out        # база меньше 100 000 - честное предупреждение

    def test_a_slow_stage_is_flagged_as_not_met(self, frames, capsys):
        slow_embed = "detect;dur=40.0, crop;dur=1.0, embed;dur=5480.0, search;dur=10.0, total;dur=5600.0"
        bench.run(parse(frames, "-n", "4", "--warmup", "1"), client=_server(self.DET, search_timing=slow_embed))

        embed_line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("Эмбеддинг"))
        assert "НЕ выполнено" in embed_line and "5480" in embed_line

    def test_warmup_requests_are_not_counted(self, frames, capsys):
        bench.run(parse(frames, "-n", "4", "--warmup", "5"), client=_server(self.DET))
        assert "Замеров: 4" in capsys.readouterr().out

    def test_frames_without_vehicles_are_skipped(self, frames, capsys):
        code = bench.run(parse(frames, "-n", "3"), client=_server([]))
        assert code == 1 and "не найдено ТС" in capsys.readouterr().err

    def test_missing_images(self, tmp_path, capsys):
        assert bench.run(parse(tmp_path / "нет"), client=_server(self.DET)) == 1

    def test_server_error_is_reported_not_raised(self, frames, capsys):
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)), base_url="http://test")
        assert bench.run(parse(frames), client=client) == 1
        assert "ОШИБКА при обращении к серверу" in capsys.readouterr().err