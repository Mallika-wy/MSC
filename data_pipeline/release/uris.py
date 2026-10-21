from pathlib import Path


def external_uri(inventory, path):
    """Give a local file an explicit root alias, shared by files in the same folder."""
    path = Path(path).resolve()
    roots = inventory.external_roots
    alias = next((key for key, value in roots.items() if value == str(path.parent)), None)
    if alias is None:
        alias = f'local_{len(roots)}'
        roots[alias] = str(path.parent)
    return f'root://{alias}/{path.name}'

