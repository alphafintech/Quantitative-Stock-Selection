"""Minimal tqdm stub used for progress display when the real package is unavailable."""
from typing import Iterable, Iterator, Optional, Any

def tqdm(iterable: Iterable, *args: Any, **kwargs: Any) -> Iterator:
    """A very small subset of tqdm that simply yields the given iterable."""
    desc = kwargs.get("desc", "")
    total = kwargs.get("total")
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except Exception:
            total = None
    for i, item in enumerate(iterable, 1):
        if total:
            print(f"{desc} {i}/{total}", end="\r", flush=True)
        yield item
    if total:
        print()
    else:
        pass
