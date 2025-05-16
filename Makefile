NETGEN_DOCUMENTATION_RST_FORMAT=1
NETGEN_DOCUMENTATION_OUT_DIR=docs
NETGEN_DOCUMENTATION_SRC_DIR=.
SPHINXOPTS?=
SPHINXBUILD?= sphinx-build -a

.PHONY: all clean_notebooks build copy_widgets clean_env

all: clean_notebooks build copy_widgets clean_env
	@echo "Done."

clean_notebooks:
	@echo "Cleaning Jupyter notebook outputs in examples/..."
	@find ./examples -name "*.ipynb" -exec jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} \;

build:
	@echo "Running sphinx-build..."
	@$(SPHINXBUILD) -a "$(NETGEN_DOCUMENTATION_SRC_DIR)" "$(NETGEN_DOCUMENTATION_OUT_DIR)" $(O)

copy_widgets:
	@echo "Copying Jupyter widgets..."
	@python3 -m webgui_jupyter_widgets.js ./_static

clean_env:
	@true  # No-op, as Makefile recipes are isolated and env vars are local
