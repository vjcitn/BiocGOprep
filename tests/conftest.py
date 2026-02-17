"""
conftest.py
-----------
Pytest configuration for build_go_db tests.
Must live in the same directory as test_build_go_db.py.

The --obo option is registered here rather than in the test module because
pytest processes command-line options before importing test files; a hook
defined inside a test module is always seen too late.
"""

def pytest_addoption(parser):
    parser.addoption(
        "--obo",
        default=None,
        metavar="PATH",
        help="Path to a real go.obo file for full-database tests.",
    )
