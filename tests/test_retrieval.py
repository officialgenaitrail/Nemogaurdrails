from retrieval import context_for, retrieve


def test_refund_policy_is_retrieved():
    assert "14 calendar days" in context_for("What is the refund period?")


def test_shipping_policy_is_retrieved():
    assert "1 to 2 business days" in context_for("How long is express shipping?")


def test_unrelated_question_has_no_match():
    assert retrieve("Explain quantum entanglement and black holes") == []


def test_empty_question_has_no_match():
    assert retrieve("  ") == []
