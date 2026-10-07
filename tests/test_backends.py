from vscell.backends import BACKENDS
from vscell.config import DEFAULT_MODELS


def test_all_four_builtin_raw_image_backends_are_registered():
    assert tuple(BACKENDS) == DEFAULT_MODELS
