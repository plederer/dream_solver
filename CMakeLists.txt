cmake_minimum_required(VERSION 3.1.0)
project(Dream-Documentation)

# Environment variables for Sphinx
set(DREAM_DOCUMENTATION_RST_FORMAT 1)
set(DREAM_DOCUMENTATION_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(DREAM_DOCUMENTATION_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/docs")

# configured documentation tools and intermediate build results
set(SPHINX_CACHE_DIR "${DREAM_DOCUMENTATION_OUT_DIR}/_doctrees")
set(SPHINX_OUTPUT_DIR ${DREAM_DOCUMENTATION_OUT_DIR})
set(SPHINX_EXECUTABLE sphinx-build -a)

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

# Build documentation using Sphinx
add_custom_target(build_docs 
    COMMAND 
    ${CMAKE_COMMAND} -E env
    NETGEN_DOCUMENTATION_RST_FORMAT=1 
    NETGEN_DOCUMENTATION_SRC_DIR=${DREAM_DOCUMENTATION_SRC_DIR}
    NETGEN_DOCUMENTATION_OUT_DIR=${DREAM_DOCUMENTATION_OUT_DIR} --
    ${SPHINX_EXECUTABLE} -a -b html -d ${SPHINX_CACHE_DIR} ${DREAM_DOCUMENTATION_SRC_DIR} ${DREAM_DOCUMENTATION_OUT_DIR}
    DEPENDS clear_docs clean_notebooks
    COMMENT "Building HTML documentation with Sphinx"
)

# Define the clean_sphinx_output target
add_custom_target(clean_sphinx_output
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${SPHINX_CACHE_DIR}
    DEPENDS build_docs
    COMMENT "Cleaning up Sphinx doctree fildes"
)

# Copy widgets
add_custom_target(copy_widgets DEPENDS build_docs
    COMMAND 
    ${CMAKE_COMMAND} -E env
    NETGEN_DOCUMENTATION_RST_FORMAT=1 --
    python3 -m webgui_jupyter_widgets.js "${DREAM_DOCUMENTATION_OUT_DIR}/_static"
    COMMAND ${CMAKE_COMMAND} -E touch "${DREAM_DOCUMENTATION_OUT_DIR}/.nojekyll"
    COMMENT "Copying Jupyter widgets and touch .nojekyll file"
)

# Group all tasks under the 'build_all' target
add_custom_target(build_all ALL DEPENDS clear_docs clean_notebooks build_docs clean_sphinx_output copy_widgets
    COMMENT "Building documentation!"
)

