import pytest

from .xputils import MOCK_NAME, XP_IS_COMPLIANT


def pytest_collection_modifyitems(config, items):
    if not XP_IS_COMPLIANT:
        mark = pytest.mark.filterwarnings(
            f"ignore:.*determine.*{MOCK_NAME}.*Array API.*"
        )
        for item in items:
            item.add_marker(mark)
