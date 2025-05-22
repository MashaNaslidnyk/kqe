# Use python3 by default; you can override by doing 'make PYTHON=python'
PYTHON ?= python3

# Always call pip as a module of your chosen Python
PIP = $(PYTHON) -m pip

BLACK_TARGETS=kqe tests experiments
ISORT_TARGETS=kqe tests experiments

.PHONY: help clean dev-install dev-install-experiments install format test

help:
	@echo "The following make targets are available:"
	@echo "	dev-install		install all dependencies for dev environment and sets a egg link to the project sources"
	@echo "	dev-install-experiments	dev-install, but also with requirements for running experiments from the original paper"
	@echo "	install			install all dependencies and the project in the current environment"
	@echo "	clean			removes egg info"
	@echo "	format			auto-format code"
	@echo "	test			run all tests"

clean:
	rm -rf *.egg-info

# --------------------------------------------------
# Automatically set ROOTDIR in kqe/local_config.py
# (in-place, no backup; works on both macOS & Linux)
# --------------------------------------------------
configure-local:
	@echo "Configuring kqe/local_config.py → ROOTDIR = $(CURDIR)"
	@perl -pi -e \
	  's|^([[:space:]]*ROOTDIR[[:space:]]*=[[:space:]]*)None$$|\1"$(CURDIR)"|' \
	  kqe/local_config.py

install: configure-local
	pip install .

dev-install: configure-local
	pip install -e ".[dev]"

galaxy_mnist:
	@git clone https://github.com/mwalmsley/galaxy_mnist.git GalaxyMNIST || true
	@cd GalaxyMNIST && $(PIP) install -e . && cd ..

dev-install-experiments: dev-install galaxy_mnist configure-local
	$(PIP) install -e ".[experiments]"

format:
	black $(BLACK_TARGETS)
	isort $(ISORT_TARGETS)

test:
	pytest tests/
