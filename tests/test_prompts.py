import pytest

from vscell.prompts import MODALITY_DESCRIPTIONS, modality_description, system_prompt
from vscell.vlm import canonical_model, provider_for


def test_mif_is_validated_whole_cell_prompt():
    text = modality_description("mIF", "whole_cell")
    assert "WHOLE-CELL" in text
    assert "Nuclei are NOT visible" in text
    assert "MEMBRANE" in text


def test_candidate_count_is_automatic():
    text = system_prompt(4, "nuclear")
    assert "FOUR candidate" in text
    assert "4 unknown methods" in text
    assert "4th=[letter]" in text


def test_preset_target_mismatch_requires_explicit_override():
    with pytest.raises(ValueError):
        modality_description("mIF", "nuclear")
    text = modality_description(
        "mIF", "nuclear", custom_description="Custom nuclear mIF description."
    )
    assert text == "Custom nuclear mIF description."


def test_all_eight_builtin_modality_descriptions_are_packaged():
    expected = {
        "DAPI": "nuclear",
        "ssDNA": "nuclear",
        "H&E": "nuclear",
        "mIF": "whole_cell",
        "phase_contrast": "whole_cell",
        "fluorescence": "whole_cell",
        "brightfield": "whole_cell",
        "DIC": "whole_cell",
    }
    assert set(MODALITY_DESCRIPTIONS) == set(expected)
    for modality, target in expected.items():
        assert modality_description(modality, target) == MODALITY_DESCRIPTIONS[modality]


def test_unknown_modality_gets_target_aware_generic_description():
    text = modality_description("spatial proteomics", "whole_cell")
    assert "spatial proteomics" in text
    assert "WHOLE-CELL" in text


def test_packaged_alternative_vlm_routes():
    assert canonical_model("gpt-6") == "gpt-6-astra"
    assert provider_for("gpt-6-astra") == "openai"
    assert canonical_model("claude") == "claude-fable-5-1"
    assert provider_for("claude-fable-5-1") == "anthropic"
