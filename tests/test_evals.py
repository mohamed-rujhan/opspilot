from opspilot.evals import load_cases, run_offline


def test_dataset_is_well_formed():
    cases = load_cases()

    assert len(cases) >= 30
    assert len({c["id"] for c in cases}) == len(cases)
    assert all(c["question"] for c in cases)


def test_offline_eval_meets_quality_bar(settings):
    report = run_offline(settings, load_cases())

    assert "Retrieval recall@3 (BM25) | 23/23" in report
    assert "Prompt injections blocked | 2/2" in report
    assert "Failures" not in report
