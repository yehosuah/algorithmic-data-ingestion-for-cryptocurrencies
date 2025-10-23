import json
import subprocess
import sys
from pathlib import Path


def test_report_shortlist_contains_calmon_candidate(tmp_path):
    """
    Run the shortlist CLI against the bundled model reports and ensure the
    deployable Calmon baseline still satisfies the release criteria.
    """
    out_path = tmp_path / "report_shortlist.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/report_shortlist.py",
            "--models-root",
            "models",
            "--out",
            str(out_path),
            "--top-n",
            "10",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out_path.exists(), "report_shortlist CLI did not materialise an output file"

    payload = json.loads(out_path.read_text())
    assert payload["total_candidates"] > 0, "Expected at least one shortlisted candidate"

    shortlisted_models = {candidate["model"] for candidate in payload["candidates"]}
    assert (
        "base_xgb_h120_calmon_spread0" in shortlisted_models
    ), "Calmon relaxed baseline should remain shortlist-worthy"

    criteria = payload["criteria"]
    min_equity = float(criteria["min_equity"])
    min_turnover = float(criteria["min_turnover"])
    require_rss = bool(criteria.get("require_rss", True))

    for candidate in payload["candidates"]:
        assert candidate["final_equity"] >= min_equity
        assert candidate["total_turnover"] >= min_turnover
        if require_rss:
            rss = candidate.get("rss_audit") or {}
            assert rss.get("passed", True), f"{candidate['model']} RSS audit regressed: {result.stdout}"
