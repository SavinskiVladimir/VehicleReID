"""
Лёгкий тест валидации аргументов CLI train_detector.py.

Сам процесс обучения (train_detector.py main()) - это тонкая обёртка
над ultralytics.YOLO.train(), уже проверенная end-to-end вручную
(реальные прогоны на синтетике и на UA-DETRAC). Модульному тестированию
здесь имеет смысл подвергать только логику, которая ДЕЙСТВИТЕЛЬНО
принадлежит этому файлу, а не ultralytics - валидацию аргументов.

Запуск: из папки ml/detection/  ->  pytest test_train_detector.py
"""

import sys

import pytest

from train_detector import parse_args


class TestParseArgs:
    def test_requires_data_or_resume(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train_detector.py"])
        with pytest.raises(SystemExit):
            parse_args()

    def test_data_alone_is_sufficient(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train_detector.py", "--data", "data.yaml"])
        args = parse_args()
        assert args.data == "data.yaml"
        assert args.resume is None

    def test_resume_alone_is_sufficient(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train_detector.py", "--resume", "last.pt"])
        args = parse_args()
        assert args.resume == "last.pt"
        assert args.data is None

    def test_default_model_is_yolo11s(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train_detector.py", "--data", "data.yaml"])
        args = parse_args()
        assert args.model == "yolo11s.pt"

    def test_default_epochs_and_imgsz(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train_detector.py", "--data", "data.yaml"])
        args = parse_args()
        assert args.epochs == 50
        assert args.imgsz == 640
