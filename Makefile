PYTHON = python3
PYTEST = $(PYTHON) -m pytest
SRC    = src
TESTS  = tests

.PHONY: test test-attacks test-controller test-utils test-view

test:
	PYTHONPATH=$(SRC) $(PYTEST) $(TESTS) -v

test-attacks:
	PYTHONPATH=$(SRC) $(PYTEST) $(TESTS)/attacks -v

test-controller:
	PYTHONPATH=$(SRC) $(PYTEST) $(TESTS)/controller -v

test-utils:
	PYTHONPATH=$(SRC) $(PYTEST) $(TESTS)/utils -v

test-view:
	PYTHONPATH=$(SRC) $(PYTEST) $(TESTS)/view -v
