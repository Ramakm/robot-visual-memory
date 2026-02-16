"""Batch processing utilities."""

from typing import Iterator, TypeVar

T = TypeVar("T")


def chunked(items: list[T], size: int) -> Iterator[list[T]]:
    """Yield successive chunks of a given size from a list.

    Args:
        items: List to chunk.
        size: Maximum chunk size.

    Yields:
        Lists of up to `size` elements.
    """
    for i in range(0, len(items), size):
        yield items[i : i + size]
