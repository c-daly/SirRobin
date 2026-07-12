import ast
from pathlib import Path


def test_gain1_generator_has_no_torch_production_or_donor_imports():
    tree = ast.parse(Path("tools/gain1_oracle.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    assert not any(name == "torch" or name.startswith("sirrobin") for name in names)
