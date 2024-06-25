from pathlib import Path
import os


def increment_path(path):
    path = Path(path)
    for n in range(1, 9999):
        p = f"{path}{n}"
        if not os.path.exists(p):
            break
    path = Path(p)
    path.mkdir(exist_ok=True)
    return path, n
