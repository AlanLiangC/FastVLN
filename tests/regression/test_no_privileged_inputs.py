import ast
from pathlib import Path


def test_policy_inputs_do_not_import_oracle_or_generate_language():
    for path in Path("src/streamnav/models").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "envs" not in (node.module or "")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "generate"
