from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_prediction_ui_does_not_present_uncalibrated_scores_as_probabilities():
    script = (PROJECT_ROOT / "app" / "static" / "js" / "main.js").read_text(encoding="utf-8")

    assert "model score (uncalibrated)" in script
    assert "not probabilities" in script
    assert "Malignant-Suspect Probability" not in script
    assert "Benign Probability" not in script
