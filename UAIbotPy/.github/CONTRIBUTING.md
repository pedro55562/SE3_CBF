# Contributing to UAIbotPy

Thank you for contributing to UAIbotPy! Follow these guidelines to ensure smooth integration of your work.

## Types of Contributions

### 1. Bug Fixes & Function Modifications

- Place C++ code in `c_implementation/` and update `pybind_main.cpp` bindings.
- Add Python functions that call the C++ equivalents.
- Raise `ImportError` with installation instructions if C++ extensions are missing (see [`Ball.projection`](../uaibot/simobjects/ball.py) for reference).
- **Testing**:
  - Run `noxfile.py` and ensure all tests pass.
  - Update/add unittests as needed.
- **Documentation**:
  - Update docstrings (NumPy-style).

### 2. New Features

- **C++ Implementation (Strongly Recommended)**:
  - Extend `c_implementation/` and update `pybind_main.cpp`.
- **Python Wrapper**:
  - Keep thin wrappers that delegate to C++.
  - Follow the same error-handling pattern as bug fixes.
  - Add comprehensive unittests and update `noxfile.py`.
- **Hybrid Approach**:
  - If providing both versions, follow [`Utils.compute_dist`](../uaibot/utils/utils.py).

### 3. New Robot Models

- **Process**:
  1. **Temporary Upload**:
     - Create a separate branch named `temp_models/[robot-name]` in your fork.
     - Add model files (OBJ, STL) to a `temp_models/[robot-name]` folder.
     - Implement DHT parameters, collision models, and joint limits (see [Kuka KR5](../uaibot/robot/_create_kuka_kr5.py) for reference).
  2. **PR Submission**:
     - Open a PR from your main contribution branch (not the `temp_models/` branch).
     - Note the temporary files in your PR description for maintainers to upload to [jsDelivr](https://www.jsdelivr.com/).
- **Important**:
  - Never merge model files directly into the main repository.
  - Maintainers will upload to jsDelivr and provide the CDN URLs for integration.

## PR Guidelines

- **Branch Naming** (recommended):
  - `fix/[issue]` for bug fixes
  - `feat/[feature]` for new features
  - `model/[robot_name]` for robot models
- **Testing Requirements**:
  - Must build on Ubuntu 22.04/24.04 (Python 3.11-3.12).
  - Should maintain compatibility with MacOS/Windows/Ubuntu (Python 3.10-3.13) where possible.
  - GitHub Actions will verify compatibility.
- **Documentation**:
  - Clearly document the C++/Python split in docstrings.
  - Update any affected examples or tutorials.
