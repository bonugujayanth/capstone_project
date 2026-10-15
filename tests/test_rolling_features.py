import pandas as pd

from add_rolling_features import add_rolling_features


def test_add_rolling_features_keeps_session_boundaries_and_label():
    df = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s1", "s1", "s2", "s2", "s2"],
            "timestamp": [
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:00:01Z",
                "2024-01-01T00:00:02Z",
                "2024-01-01T00:00:03Z",
                "2024-01-01T00:00:10Z",
                "2024-01-01T00:00:11Z",
                "2024-01-01T00:00:12Z",
            ],
            "write_bytes_per_s": [10, 20, 30, 40, 3, 6, 9],
            "cpu_pct": [10, 20, 30, 40, 1, 2, 3],
            "files_modified": [1, 1, 2, 3, 0, 1, 0],
            "label": [0, 0, 1, 1, 0, 0, 1],
        }
    )

    out = add_rolling_features(df, windows=[5], base_features=["write_bytes_per_s", "cpu_pct", "files_modified"])

    assert "write_bytes_per_s_mean_5s" in out.columns
    assert "cpu_pct_std_5s" in out.columns
    assert "files_modified_sum_5s" in out.columns
    assert "label" in out.columns
    assert out["session_id"].nunique() == 2
    assert out["write_bytes_per_s_mean_5s"].isna().sum() == 0
    assert out["write_bytes_per_s_delta_5s"].notna().all()
