import pytest

from vscell.ranking import borda_aggregate, parse_ranking, shuffled_mapping


def test_shuffle_is_stable_and_changes_by_repetition():
    models = ["a", "b", "c", "d"]
    assert shuffled_mapping("crop", 0, models) == shuffled_mapping("crop", 0, models)
    assert shuffled_mapping("crop", 0, models) != shuffled_mapping("crop", 1, models)


def test_parse_and_borda():
    mapping = {"A": "m3", "B": "m1", "C": "m4", "D": "m2"}
    order = parse_ranking(
        "Assessment here\nRANKING: 1st=B 2nd=[D] 3rd=A 4th=C", mapping
    )
    assert order == ["m1", "m2", "m3", "m4"]
    final, scores = borda_aggregate(
        [
            ["m1", "m2", "m3", "m4"],
            ["m2", "m1", "m3", "m4"],
            ["m1", "m3", "m2", "m4"],
        ]
    )
    assert final[0] == "m1"
    assert final[-1] == "m4"
    assert scores["m1"] == 11


def test_parse_rejects_repeats():
    mapping = {"A": "a", "B": "b", "C": "c", "D": "d"}
    with pytest.raises(ValueError):
        parse_ranking("RANKING: 1st=A 2nd=A 3rd=C 4th=D", mapping)
