"""
Тесты для prepare_dataset.py (конвертация UA-DETRAC -> YOLO-формат).

Запуск: из корня проекта  ->  pytest tests/test_prepare_dataset.py
(требуется pytest.ini в корне проекта с pythonpath, включающим ml/detection/)
"""

import os
import subprocess
import sys

import pytest
import yaml
from PIL import Image

import prepare_dataset
from prepare_dataset import _boxes_to_yolo_lines, _link_or_copy, _parse_sequence_xml


# ---------------------------------------------------------------------------
# _parse_sequence_xml
# ---------------------------------------------------------------------------

_SAMPLE_XML = """<sequence name="MVI_20011">
  <frame num="1">
    <target_list>
      <target id="1">
        <box left="10" top="10" width="40" height="20"/>
        <attribute vehicle_type="car" occlusion_ratio="0.0" truncation_ratio="0.0"/>
      </target>
      <target id="2">
        <box left="100" top="50" width="60" height="30"/>
        <attribute vehicle_type="bus" occlusion_ratio="0.9" truncation_ratio="0.0"/>
      </target>
    </target_list>
  </frame>
  <frame num="2">
    <target_list>
      <target id="1">
        <box left="15" top="12" width="40" height="20"/>
        <attribute vehicle_type="van" occlusion_ratio="0.0" truncation_ratio="0.5"/>
      </target>
    </target_list>
  </frame>
  <frame num="3">
    <target_list />
  </frame>
</sequence>
"""


class TestParseSequenceXml:
    def test_parses_all_frames(self, tmp_path):
        xml_path = tmp_path / "MVI_20011.xml"
        xml_path.write_text(_SAMPLE_XML)

        frames = _parse_sequence_xml(str(xml_path), max_occlusion_ratio=1.0, max_truncation_ratio=1.0)

        assert set(frames.keys()) == {1, 2, 3}
        assert len(frames[1]) == 2
        assert len(frames[2]) == 1
        assert len(frames[3]) == 0

    def test_filters_by_occlusion_ratio(self, tmp_path):
        xml_path = tmp_path / "MVI_20011.xml"
        xml_path.write_text(_SAMPLE_XML)

        frames = _parse_sequence_xml(str(xml_path), max_occlusion_ratio=0.8, max_truncation_ratio=1.0)
        # На frame 1 второй target (bus, occlusion=0.9) должен быть отфильтрован
        assert len(frames[1]) == 1

    def test_filters_by_truncation_ratio(self, tmp_path):
        xml_path = tmp_path / "MVI_20011.xml"
        xml_path.write_text(_SAMPLE_XML)

        frames = _parse_sequence_xml(str(xml_path), max_occlusion_ratio=1.0, max_truncation_ratio=0.3)
        # На frame 2 единственный target имеет truncation_ratio=0.5 -> отфильтрован
        assert len(frames[2]) == 0

    def test_box_values_are_correct(self, tmp_path):
        xml_path = tmp_path / "MVI_20011.xml"
        xml_path.write_text(_SAMPLE_XML)

        frames = _parse_sequence_xml(str(xml_path), max_occlusion_ratio=1.0, max_truncation_ratio=1.0)
        left, top, width, height = frames[1][0]
        assert (left, top, width, height) == (10.0, 10.0, 40.0, 20.0)


# ---------------------------------------------------------------------------
# _boxes_to_yolo_lines
# ---------------------------------------------------------------------------

class TestBoxesToYoloLines:
    def test_correct_normalization(self):
        # left=10, top=10, width=40, height=20, кадр 200x100
        boxes = [(10.0, 10.0, 40.0, 20.0)]
        lines = _boxes_to_yolo_lines(boxes, img_w=200, img_h=100)

        assert len(lines) == 1
        class_id, x_c, y_c, w, h = lines[0].split()
        assert class_id == "0"
        assert float(x_c) == pytest.approx(0.15)
        assert float(y_c) == pytest.approx(0.20)
        assert float(w) == pytest.approx(0.20)
        assert float(h) == pytest.approx(0.20)

    def test_clamps_out_of_bounds_box(self):
        # Бокс частично выходит за пределы кадра (например, из-за
        # погрешности разметки на краю кадра).
        boxes = [(-5.0, -5.0, 50.0, 50.0)]
        lines = _boxes_to_yolo_lines(boxes, img_w=100, img_h=100)

        assert len(lines) == 1
        _, x_c, y_c, w, h = lines[0].split()
        for value in (x_c, y_c, w, h):
            assert 0.0 <= float(value) <= 1.0

    def test_skips_degenerate_box(self):
        boxes = [(10.0, 10.0, 0.0, 20.0)]  # width=0 - вырожденный бокс
        lines = _boxes_to_yolo_lines(boxes, img_w=100, img_h=100)
        assert lines == []

    def test_multiple_boxes(self):
        boxes = [(0.0, 0.0, 20.0, 20.0), (50.0, 50.0, 20.0, 20.0)]
        lines = _boxes_to_yolo_lines(boxes, img_w=100, img_h=100)
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# _link_or_copy
# ---------------------------------------------------------------------------

class TestLinkOrCopy:
    def test_copy_mode_creates_real_file(self, tmp_path):
        src = tmp_path / "src.jpg"
        Image.new("RGB", (8, 8)).save(src)
        dst = tmp_path / "dst.jpg"

        _link_or_copy(str(src), str(dst), mode="copy")

        assert dst.is_file()
        assert not dst.is_symlink()

    def test_symlink_mode_creates_link_or_falls_back_to_copy(self, tmp_path):
        src = tmp_path / "src.jpg"
        Image.new("RGB", (8, 8)).save(src)
        dst = tmp_path / "dst.jpg"

        _link_or_copy(str(src), str(dst), mode="symlink")

        # На системах без прав на symlink функция сама переключается на
        # copy - в обоих случаях итоговый файл должен существовать и
        # быть читаемым.
        assert dst.is_file()

    def test_does_not_overwrite_existing_file(self, tmp_path):
        src = tmp_path / "src.jpg"
        Image.new("RGB", (8, 8)).save(src)
        dst = tmp_path / "dst.jpg"
        dst.write_bytes(b"already here")

        _link_or_copy(str(src), str(dst), mode="copy")
        assert dst.read_bytes() == b"already here"


# ---------------------------------------------------------------------------
# main() - интеграционный прогон конвертации целиком
# ---------------------------------------------------------------------------

def _build_fake_detrac(root, num_sequences=4, frames_per_seq=3, img_size=(200, 100)):
    img_root = os.path.join(root, "Insight-MVT_Annotation_Train")
    ann_root = os.path.join(root, "DETRAC-Train-Annotations-XML")
    os.makedirs(img_root, exist_ok=True)
    os.makedirs(ann_root, exist_ok=True)

    for i in range(num_sequences):
        seq_name = f"MVI_2001{i}"
        seq_dir = os.path.join(img_root, seq_name)
        os.makedirs(seq_dir, exist_ok=True)

        frame_tags = []
        for frame_num in range(1, frames_per_seq + 1):
            Image.new("RGB", img_size, (100, 100, 100)).save(
                os.path.join(seq_dir, f"img{frame_num:05d}.jpg")
            )
            frame_tags.append(
                f'<frame num="{frame_num}"><target_list><target id="1">'
                f'<box left="10" top="10" width="40" height="20"/>'
                f'<attribute vehicle_type="car" occlusion_ratio="0.0" truncation_ratio="0.0"/>'
                f'</target></target_list></frame>'
            )

        xml_content = f'<sequence name="{seq_name}">' + "".join(frame_tags) + "</sequence>"
        with open(os.path.join(ann_root, f"{seq_name}.xml"), "w") as f:
            f.write(xml_content)


class TestMainIntegration:
    def test_end_to_end_conversion(self, tmp_path):
        detrac_root = tmp_path / "UA-DETRAC"
        output_root = tmp_path / "yolo_out"
        _build_fake_detrac(str(detrac_root), num_sequences=4, frames_per_seq=3)

        script_path = prepare_dataset.__file__  # реальный путь к модулю, независимо от расположения теста
        result = subprocess.run(
            [
                sys.executable, script_path,
                "--detrac-root", str(detrac_root),
                "--output", str(output_root),
                "--val-fraction", "0.5",
                "--link-mode", "copy",
                "--seed", "1",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        data_yaml_path = output_root / "data.yaml"
        assert data_yaml_path.is_file()

        with open(data_yaml_path) as f:
            data_yaml = yaml.safe_load(f)
        assert data_yaml["names"] == {0: "vehicle"}

        train_images = list((output_root / "images" / "train").glob("*.jpg"))
        val_images = list((output_root / "images" / "val").glob("*.jpg"))
        assert len(train_images) + len(val_images) == 4 * 3  # 4 seq x 3 кадра

        # Для каждого изображения должен существовать соответствующий label
        for img_path in train_images + val_images:
            label_path = (
                output_root / "labels" /
                img_path.parent.name / (img_path.stem + ".txt")
            )
            assert label_path.is_file()

    def test_max_sequences_limits_processed_sequences(self, tmp_path):
        detrac_root = tmp_path / "UA-DETRAC"
        output_root = tmp_path / "yolo_out"
        _build_fake_detrac(str(detrac_root), num_sequences=6, frames_per_seq=2)

        script_path = prepare_dataset.__file__  # реальный путь к модулю, независимо от расположения теста
        result = subprocess.run(
            [
                sys.executable, script_path,
                "--detrac-root", str(detrac_root),
                "--output", str(output_root),
                "--max-sequences", "2",
                "--val-fraction", "0.5",
                "--link-mode", "copy",
                "--seed", "1",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        train_images = list((output_root / "images" / "train").glob("*.jpg"))
        val_images = list((output_root / "images" / "val").glob("*.jpg"))
        # 2 последовательности x 2 кадра = 4 изображения всего
        assert len(train_images) + len(val_images) == 4

    def test_frame_stride_reduces_frame_count(self, tmp_path):
        detrac_root = tmp_path / "UA-DETRAC"
        output_root = tmp_path / "yolo_out"
        _build_fake_detrac(str(detrac_root), num_sequences=1, frames_per_seq=6)

        script_path = prepare_dataset.__file__  # реальный путь к модулю, независимо от расположения теста
        result = subprocess.run(
            [
                sys.executable, script_path,
                "--detrac-root", str(detrac_root),
                "--output", str(output_root),
                "--frame-stride", "2",
                "--val-fraction", "0",
                "--link-mode", "copy",
                "--seed", "1",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        all_images = list((output_root / "images").rglob("*.jpg"))
        # 6 кадров, stride=2 -> кадры 1,3,5 -> 3 изображения
        assert len(all_images) == 3