import numpy as np

from vscell.ensemble import select_solution08, vscell_consensus


def _mask(anchor=True, residual=True):
    result = np.zeros((20, 20), dtype=np.int32)
    if anchor:
        result[2:7, 2:7] = 1
    if residual:
        result[12:17, 12:17] = 2
    return result


def test_solution08_keeps_supported_anchor_and_adds_peer_consensus():
    predictions = {
        "anchor": _mask(anchor=True, residual=False),
        "peer1": _mask(),
        "peer2": _mask(),
        "peer3": _mask(anchor=False, residual=True),
    }
    ranking = ["anchor", "peer1", "peer2", "peer3"]
    selected, audit = select_solution08(predictions, ranking)
    assert int(selected.max()) == 2
    assert audit["anchor_model"] == "anchor"
    assert audit["actions"]["new_over_null"] == 1
    final, final_audit = vscell_consensus(predictions, ranking)
    assert int(final.max()) == 2
    assert final_audit["selected_cells_after_refinement"] == 2


def test_solution08_is_gt_free_signature():
    predictions = {name: _mask() for name in ("a", "b", "c", "d")}
    final, _ = vscell_consensus(predictions, ["a", "b", "c", "d"])
    assert final.shape == (20, 20)
