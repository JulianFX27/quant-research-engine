.PHONY: smoke test

smoke:
	python -m pip install -q pytest
	pytest -q

test:
	pytest

