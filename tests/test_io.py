import numpy as np

from vscell.io import relabel_instances


def test_relabel_splits_disconnected_islands_and_keeps_diagonal_connection():
    mask = np.zeros((8, 8), dtype=np.int32)
    mask[1, 1] = 7
    mask[2, 2] = 7  # diagonally connected under 8-connectivity
    mask[6, 6] = 7  # disconnected island with the same source label
    mask[1:3, 5:7] = 12

    result = relabel_instances(mask)

    assert np.unique(result).tolist() == [0, 1, 2, 3]
    assert result[1, 1] == result[2, 2]
    assert result[6, 6] != result[1, 1]
    assert len(np.unique(result[1:3, 5:7])) == 1
