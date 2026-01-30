.PHONY: smoke test

PYTEST ?= pytest

smoke:
	$(PYTEST) -q

test:
	$(PYTEST)
