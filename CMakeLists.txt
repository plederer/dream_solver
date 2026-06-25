cmake_minimum_required(VERSION 3.1...4.0)
project(Dream-Documentation)

# Environment variables for Sphinx
set(DREAM_DOCUMENTATION_RST_FORMAT 1)
set(DREAM_DOCUMENTATION_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(DREAM_DOCUMENTATION_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/docs")

# configured documentation tools and intermediate build results
set(SPHINX_CACHE_DIR "${DREAM_DOCUMENTATION_OUT_DIR}/_doctrees")
set(SPHINX_OUTPUT_DIR ${DREAM_DOCUMENTATION_OUT_DIR})
set(SPHINX_EXECUTABLE sphinx-build)

# Clear previous documentation
add_custom_target(clear_docs
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${DREAM_DOCUMENTATION_OUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DREAM_DOCUMENTATION_OUT_DIR}
    COMMENT "Cleaning up old documentation files"
)

# Recursively collect all .ipynb files in the examples directory
file(GLOB_RECURSE NOTEBOOK_FILES RELATIVE "${DREAM_DOCUMENTATION_SRC_DIR}" "${DREAM_DOCUMENTATION_SRC_DIR}/examples/*.ipynb")

# Build up the custom command list
set(NOTEBOOK_CLEAN_COMMANDS "")
foreach(NOTEBOOK ${NOTEBOOK_FILES})
    list(APPEND NOTEBOOK_CLEAN_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E echo "Clearing output in ${NOTEBOOK}"
        COMMAND jupyter nbconvert --clear-output --inplace "${DREAM_DOCUMENTATION_SRC_DIR}/${NOTEBOOK}"
    )
endforeach()

# Define the clean_notebooks target
add_custom_target(clean_notebooks
    ${NOTEBOOK_CLEAN_COMMANDS}
    COMMENT "Cleaning up Jupyter notebook outputs"
)

# Build documentation from scratch using Sphinx: wipes the output directory and the Sphinx
# doctree cache beforehand (via clear_docs) and clears all notebook outputs (via
# clean_notebooks), so every example notebook is re-executed. Use this target whenever the
# example notebooks themselves (or their expected output) have changed.
add_custom_target(build_docs
    COMMAND
    ${CMAKE_COMMAND} -E env
    NETGEN_DOCUMENTATION_RST_FORMAT=1
    NETGEN_DOCUMENTATION_SRC_DIR=${DREAM_DOCUMENTATION_SRC_DIR}
    NETGEN_DOCUMENTATION_OUT_DIR=${DREAM_DOCUMENTATION_OUT_DIR} --
    ${SPHINX_EXECUTABLE} -a -b html -d ${SPHINX_CACHE_DIR} ${DREAM_DOCUMENTATION_SRC_DIR} ${DREAM_DOCUMENTATION_OUT_DIR}
    DEPENDS clear_docs clean_notebooks
    COMMENT "Building HTML documentation with Sphinx (full rebuild, re-runs all example notebooks)"
)

# Update the documentation incrementally: keeps the existing output directory and the Sphinx
# doctree cache, and does not clear/re-run the example notebooks. Only documents whose sources
# actually changed (e.g. updated .rst/.md pages, or docstrings picked up by autodoc) are
# rebuilt; unchanged notebooks are left untouched. Use this target when only text, docstrings,
# or other non-notebook content was updated.
#
# Note: the very first run after a full build_all (which deletes the doctree cache via
# clean_sphinx_output) will still rebuild everything once, since there is no cache yet to
# compare against. Subsequent update_docs runs will then skip unchanged notebooks.
add_custom_target(update_docs
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DREAM_DOCUMENTATION_OUT_DIR}
    COMMAND
    ${CMAKE_COMMAND} -E env
    NETGEN_DOCUMENTATION_RST_FORMAT=1
    NETGEN_DOCUMENTATION_SRC_DIR=${DREAM_DOCUMENTATION_SRC_DIR}
    NETGEN_DOCUMENTATION_OUT_DIR=${DREAM_DOCUMENTATION_OUT_DIR} --
    ${SPHINX_EXECUTABLE} -b html -d ${SPHINX_CACHE_DIR} ${DREAM_DOCUMENTATION_SRC_DIR} ${DREAM_DOCUMENTATION_OUT_DIR}
    # Sphinx skips re-copying html_static_path assets entirely when no document is outdated,
    # so static-only changes (e.g. custom.css) would otherwise silently not propagate. Always
    # re-sync it explicitly.
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${DREAM_DOCUMENTATION_SRC_DIR}/_static" "${DREAM_DOCUMENTATION_OUT_DIR}/_static"
    COMMENT "Updating HTML documentation with Sphinx (incremental, keeps existing example output)"
)

# Define the clean_sphinx_output target. Only used after a full build_all, since update_docs
# relies on the doctree cache surviving between runs to skip unchanged notebooks.
add_custom_target(clean_sphinx_output
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${SPHINX_CACHE_DIR}
    DEPENDS build_docs
    COMMENT "Cleaning up Sphinx doctree files"
)

# Copy widgets (full rebuild path)
add_custom_target(copy_widgets DEPENDS build_docs
    COMMAND
    ${CMAKE_COMMAND} -E env
    NETGEN_DOCUMENTATION_RST_FORMAT=1 --
    python3 -m webgui_jupyter_widgets.js "${DREAM_DOCUMENTATION_OUT_DIR}/_static"
    COMMAND ${CMAKE_COMMAND} -E touch "${DREAM_DOCUMENTATION_OUT_DIR}/.nojekyll"
    COMMENT "Copying Jupyter widgets and touch .nojekyll file"
)

# Copy widgets (incremental update path), kept as a separate target from copy_widgets so that
# update_all never pulls in the full, notebook-clearing build_docs/clear_docs chain.
add_custom_target(copy_widgets_update DEPENDS update_docs
    COMMAND
    ${CMAKE_COMMAND} -E env
    NETGEN_DOCUMENTATION_RST_FORMAT=1 --
    python3 -m webgui_jupyter_widgets.js "${DREAM_DOCUMENTATION_OUT_DIR}/_static"
    COMMAND ${CMAKE_COMMAND} -E touch "${DREAM_DOCUMENTATION_OUT_DIR}/.nojekyll"
    COMMENT "Copying Jupyter widgets and touch .nojekyll file"
)

# Group all tasks under the 'build_all' target: full rebuild, re-runs all example notebooks.
add_custom_target(build_all ALL DEPENDS clear_docs clean_notebooks build_docs clean_sphinx_output copy_widgets
    COMMENT "Building documentation from scratch!"
)

# Group the incremental update under the 'update_all' target: keeps existing example output
# and the doctree cache, only rebuilds changed documents, and never touches/re-runs notebooks.
add_custom_target(update_all DEPENDS update_docs copy_widgets_update
    COMMENT "Updating documentation incrementally!"
)
