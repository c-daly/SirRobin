from pathlib import Path

from sirrobin.validation.corpus import EXPECTED_HISTOGRAMS, load_corpus, verify_sidecar

CORPUS = Path("oracle/fixtures/corpus.json")


def test_corpus_is_frozen_and_has_exact_authorization_content():
    data = load_corpus(CORPUS)
    assert data["classes"] == {"H0": 64, "H1": 64, "H2": 64}
    assert verify_sidecar(CORPUS)
    for name, histogram in EXPECTED_HISTOGRAMS.items():
        assert sum(histogram.values()) == 64, name
