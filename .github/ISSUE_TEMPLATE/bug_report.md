---
name: Bug Report
about: Create a report to help us improve NanoInfer
title: '[Bug] '
labels: bug
assignees: ''
---

## Environment

- **OS:** Linux (e.g., Ubuntu 22.04)
- **NanoInfer version:** (e.g., v0.5.0, commit hash)
- **CUDA version:** (e.g., 12.4)
- **Python version:** (e.g., 3.10)
- **Compiler:** (e.g., GCC 11.4)

## Build configuration

```bash
# Provide the CMake command used, if applicable
cmake -B build -DCMAKE_BUILD_TYPE=Release ...
```

## Description

A clear and concise description of the bug.

## Reproduction steps

1. ...
2. ...
3. ...

## Expected behavior

What did you expect to happen?

## Actual behavior

What actually happened? Include error messages, logs, or stack traces.

```
(paste logs here)
```

## Minimal reproducer

If possible, provide a minimal Python script or CLI command that reproduces the issue.

```python
import nanoinfer
# ...
```

## Additional context

- Are you using CPU or CUDA backend?
- Model file being used (if applicable):
- Any other relevant information:
