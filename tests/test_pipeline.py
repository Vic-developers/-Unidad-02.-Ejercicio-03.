import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nlp_project import clean_for_split, normalize_text, read_corpus, run_experiment, validate_data


def test_normalize_duplicate_key():
    assert normalize_text("  Hello\nWORLD ") == "hello world"


def test_rejects_missing_column():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(pd.DataFrame({"text": ["hi"]}))


def test_removes_repeated_text_before_split():
    df = pd.DataFrame({"text": ["same", " SAME ", "other", "more"],
                       "label": ["ham", "ham", "spam", "spam"]})
    cleaned, summary = clean_for_split(df)
    assert len(cleaned) == 3
    assert summary["duplicate_rows_removed"] == 1


def test_conflicting_duplicates_are_removed_as_groups():
    df = pd.DataFrame({"text": ["same", " SAME ", "other", "more"],
                       "label": ["ham", "spam", "ham", "spam"]})
    cleaned, summary = clean_for_split(df)
    assert len(cleaned) == 2
    assert summary["conflicting_rows_removed"] == 2


def test_official_archive_has_expected_schema_and_labels():
    df = read_corpus()
    assert set(df.columns) == {"text", "label"}
    assert set(df.label.unique()) == {"ham", "spam"}
    assert len(df) > 5000
