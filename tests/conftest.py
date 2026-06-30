"""pytest configuration for the py-icare test suite.

Its presence ensures the ``tests/`` directory is importable, so test modules can
``from icare_test_utils import ...`` regardless of the working directory pytest
is launched from. Shared helpers, paths, and tolerances live in
``tests/icare_test_utils.py``; the ``slow`` marker is registered in
``pyproject.toml``.
"""
