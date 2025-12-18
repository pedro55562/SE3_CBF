import nox
import os

def get_python_versions():
    # Get version from environment variable or use default list
    gh_python = os.environ.get("GITHUB_PYTHON_VERSION")
    if gh_python:
        return [gh_python]
    return ["3.10", "3.11", "3.12", "3.13"]  # Default versions for local development

@nox.session(python=get_python_versions())
def tests(session: nox.Session) -> None:
    """Build the package."""
    session.install("cmake>=3.18", "setuptools", "cmake", "wheel")  # Build tools first
    session.install("pytest",)
    session.install(".")
    # session.run("pip", "install", ".", "--use-pep517")
    # Run all utils tests
    session.run(
        "pytest",
        "tests/test_utils/",
        "tests/test_simobjects/",
        "--tb=long",   # Full error tracebacks
        "--color=yes", # Colorized output
    )
    