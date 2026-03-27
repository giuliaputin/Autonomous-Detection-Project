.PHONY: help install install-dev run-webcam run-decode-live run-image-debug run-image-debug-fallback train-detector lint format test pre-commit clean

help:
	@echo "Targets:"
	@echo "  make install            Install runtime dependencies"
	@echo "  make install-dev        Install runtime + dev dependencies"
	@echo "  make run-webcam         Start webcam-only mode"
	@echo "  make run-decode-live    Start live detect+decode mode"
	@echo "  make run-image-debug    Run image debug mode on qr_decoder/data"
	@echo "  make run-image-debug-fallback  Run image debug with full-frame decode fallback"
	@echo "  make train-detector     Start detector training entrypoint"
	@echo "  make lint               Run ruff checks"
	@echo "  make format             Auto-format code"
	@echo "  make test               Run pytest test suite"
	@echo "  make pre-commit         Run pre-commit hooks"
	@echo "  make clean              Remove Python cache files"

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements.txt
	python -m pip install -e .[dev]

run-webcam:
	python main.py webcam

run-decode-live:
	python main.py live --model qr_detection/runs/detect/qr_detector_v17/weights/best.pt --fps 5 --log-level INFO

run-image-debug:
	python main.py image --input qr_decoder/data --log-level DEBUG --show

run-image-debug-fallback:
	python main.py image --input qr_decoder/data --log-level DEBUG --full-frame-fallback --show

train-detector:
	python -m qr_detection.train

lint:
	python -m ruff check .
	python -m ruff format --check .

format:
	python -m ruff check --fix .
	python -m ruff format .

test:
	python -m pytest tests -v

pre-commit:
	python -m pre_commit run --all-files

clean:
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
	python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
