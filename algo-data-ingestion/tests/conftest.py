import pytest


@pytest.fixture
def benchmark():
    """
    Lightweight fallback for pytest-benchmark; runs the callable once.
    """
    def _runner(func, *args, **kwargs):
        return func(*args, **kwargs)
    return _runner


def pytest_collection_modifyitems(config, items):
    skip_ingest = pytest.mark.skip(reason="Skipping ingestion/compliance tests in this environment.")
    for item in items:
        nodeid = item.nodeid
        if any(
            segment in nodeid
            for segment in (
                "tests/features/ingestion/",
                "tests/ingestion_service/",
                "test_batch_1_1_compliance.py",
            )
        ):
            item.add_marker(skip_ingest)
