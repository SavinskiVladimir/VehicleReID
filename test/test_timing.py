"""Тесты StageTimer (backend/app/timing.py): поэтапные замеры времени запроса, C3 плана."""

import re
import time

import pytest

from backend.app.timing import StageTimer


class TestStageTimer:
    def test_measures_a_block_in_milliseconds(self):
        timer = StageTimer()
        with timer.stage("work"):
            time.sleep(0.03)
        assert 25 <= timer.stages["work"] < 500

    def test_repeated_stage_name_accumulates(self):
        timer = StageTimer()
        for _ in range(2):
            with timer.stage("work"):
                time.sleep(0.02)
        assert timer.stages["work"] >= 35

    def test_time_is_recorded_even_when_the_block_fails(self):
        timer = StageTimer()
        with pytest.raises(RuntimeError):
            with timer.stage("broken"):
                time.sleep(0.02)
                raise RuntimeError("сбой")
        assert timer.stages["broken"] >= 15

    def test_total_covers_the_sum_of_sequential_stages(self):
        timer = StageTimer()
        with timer.stage("a"):
            time.sleep(0.01)
        with timer.stage("b"):
            time.sleep(0.01)
        assert timer.total_ms >= sum(timer.stages.values())

    def test_server_timing_header_format(self):
        timer = StageTimer()
        with timer.stage("detect"):
            pass
        with timer.stage("embed"):
            pass
        # формат стандарта Server-Timing: имя;dur=миллисекунды, через запятую
        assert re.fullmatch(r"detect;dur=\d+\.\d, embed;dur=\d+\.\d, total;dur=\d+\.\d", timer.server_timing())

    def test_summary_for_the_log(self):
        timer = StageTimer()
        with timer.stage("detect"):
            pass
        assert re.fullmatch(r"detect=\d+мс total=\d+мс", timer.summary())

    def test_no_stages_still_reports_total(self):
        assert re.fullmatch(r"total;dur=\d+\.\d", StageTimer().server_timing())