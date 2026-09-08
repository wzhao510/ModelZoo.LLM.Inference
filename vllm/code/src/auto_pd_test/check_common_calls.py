import ast
import inspect
import pathlib
import sys

sys.path.insert(0, ".")
import common  # noqa: E402

root = pathlib.Path(".")

# name -> real function object (only the ones tests actually import from common)
REAL_FUNCS = {
    n: getattr(common, n)
    for n in dir(common)
    if not n.startswith("_") and inspect.isfunction(getattr(common, n))
}

problems = []

for path in sorted(root.glob("*/test_*.py")):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    # local_name -> real common.py name, for `from common import X as Y`
    alias_map = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "common":
            for a in node.names:
                alias_map[a.asname or a.name] = a.name

    if not alias_map:
        continue

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            local_name = node.func.id
            if local_name not in alias_map:
                continue
            real_name = alias_map[local_name]
            func = REAL_FUNCS.get(real_name)
            if func is None:
                problems.append((str(path), node.lineno, local_name, f"'{real_name}' not found in common.py"))
                continue
            sig = inspect.signature(func)
            params = sig.parameters
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            for kw in node.keywords:
                if kw.arg is None:
                    continue  # **kwargs at call site, can't check
                if kw.arg not in params and not has_var_kw:
                    problems.append((
                        str(path), node.lineno, local_name,
                        f"unexpected keyword '{kw.arg}' (real params: {list(params)})",
                    ))
            # positional arg count sanity check (best-effort; skips *args/**kwargs-using calls)
            positional_params = [
                p for p in params.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            n_pos_args = sum(1 for a in node.args if not isinstance(a, ast.Starred))
            has_star_args = any(isinstance(a, ast.Starred) for a in node.args)
            if not has_star_args and n_pos_args > len(positional_params) and not has_var_kw:
                problems.append((
                    str(path), node.lineno, local_name,
                    f"{n_pos_args} positional args but only {len(positional_params)} positional params exist",
                ))

if problems:
    print(f"FOUND {len(problems)} CALL-SITE MISMATCH(ES):")
    for p in problems:
        print(" ", p)
    sys.exit(1)
else:
    print("No call-site mismatches found against common.py's real signatures.")
