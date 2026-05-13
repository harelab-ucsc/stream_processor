cmake_minimum_required(VERSION 3.14)
project(stream_processor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11_vendor QUIET)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)

# ── pybind11 C++ extension ────────────────────────────────────────────────────
pybind11_add_module(_core MODULE
  src/spectral_correct/process.cpp
  src/spectral_correct/bindings.cpp
)
target_include_directories(_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/spectral_correct
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(_core PRIVATE ${OpenCV_LIBS})

# ── Python package install ────────────────────────────────────────────────────
ament_python_install_package(${PROJECT_NAME})

# Install the .so inside the spectral_correct sub-package directory so that
# `from stream_processor.spectral_correct import process_cam0` resolves
# `_core.<abi>.so` next to its __init__.py.
install(TARGETS _core
  LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}/spectral_correct"
)

# ── Console script (replaces setup.py entry_point) ────────────────────────────
install(PROGRAMS scripts/sync_node
  DESTINATION lib/${PROJECT_NAME}
)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
