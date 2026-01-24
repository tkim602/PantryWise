import ast

def parse_list(cell):
    """Convert a CSV cell that looks like a Python list string into a real list."""
    if isinstance(cell, list):
        return cell
    if not isinstance(cell, str) or not cell:
        return []
    try:
        value = ast.literal_eval(cell)
        if isinstance(value, list):
            return value
        return [str(value)]
    except Exception:
        return [c.strip() for c in str(cell).split(",") if c.strip()]
