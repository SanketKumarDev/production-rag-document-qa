from app.evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"b"}, 2) == 1.0


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], {"b"}, 2) == 0.5


def test_mrr():
    assert reciprocal_rank(["x", "b"], {"b"}) == 0.5
