# Variables
PROJECT = aintelope
TESTS = tests
VENV = venv_$(PROJECT)

run-training-short:
	python -m ${PROJECT}

run-training-long:
	echo 'run-training-long currently not implemented'

.PHONY: venv
venv: ## create virtual environment
	@if [ ! -f "$(VENV)/bin/activate" ]; then python3 -m venv $(VENV) ; fi;

.PHONY: clean-venv
clean-venv: ## remove virtual environment
	if [ -d $(VENV) ]; then rm -r $(VENV) ; fi;

.PHONY: install
install: ## Install packages
	pip install -r requirements/$(PROJECT).txt

.PHONY: install-dev
install-dev: ## Install development packages
	pip install -r requirements/$(PROJECT)_dev.txt

.PHONY: install-all
install-all: install install-dev ## install all packages

.PHONY: build-local
build-local: ## install the project locally
	pip install -e .

.PHONY: tests-local
tests-local: ## Run tests locally
	python -m pytest --cov=$(PROJECT) $(TESTS)

.PHONY: typecheck-local
typecheck-local: ## Local typechecking
	mypy $(PROJECT)

.PHONY: isort
isort: ## Sort python imports
	isort .

.PHONY: clean
clean:
	rm -rf *.egg-info
	rm -rf .mypy_cache
	rm -rf .pytest_cache

.PHONY: help
help: ## Show this help
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)
