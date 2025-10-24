.PHONY: check format lint clean

check:
	python -m compileall scripts/train_subtitles_transformer.py dashboard/server.py

format:
	autopep8 --in-place --aggressive --recursive scripts || true

lint:
	flake8 scripts || true

clean:
	rm -rf __pycache__ .pytest_cache
