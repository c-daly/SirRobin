import json
from pathlib import Path

from sirrobin.benchmarks.locomotion import benchmark_cell


def test_benchmark_counts_only_live_scientific_work():
    corpus = json.loads(Path("oracle/fixtures/corpus.json").read_text())
    result = benchmark_cell(corpus, "H1", capacity=8, live=6, device="cpu", steps=2, warmup=1, repetitions=2)
    assert result.status == "ok"
    assert result.live_bodies == 6
    assert len(result.repetitions) == 2
    assert result.minimum > 0
