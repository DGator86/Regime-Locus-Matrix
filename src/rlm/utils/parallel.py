"""Parallel execution helpers with optional Ray backend."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def _normalize_max_workers(max_workers: int | None) -> int:
    if max_workers is None:
        return 1
    return max(1, int(max_workers))


def _run_serial(items: list[T], fn: Callable[[T], R]) -> list[R]:
    return [fn(item) for item in items]


def _run_executor(
    items: list[T],
    fn: Callable[[T], R],
    *,
    max_workers: int,
    use_threads: bool,
) -> list[R]:
    results: list[R | None] = [None] * len(items)
    pool_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with pool_cls(max_workers=max_workers) as pool:
        future_to_idx: dict[Future[R], int] = {
            pool.submit(fn, item): idx for idx, item in enumerate(items)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
    return [r for r in results if r is not None]


def _run_ray(items: list[T], fn: Callable[[T], R], *, max_workers: int) -> list[R]:
    try:
        import ray
    except Exception:
        return _run_executor(items, fn, max_workers=max_workers, use_threads=False)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=max_workers)

    @ray.remote
    def _call(item: T) -> R:
        return fn(item)

    refs = [_call.remote(item) for item in items]
    return list(ray.get(refs))


def parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int | None = None,
    backend: str = "process",
) -> list[R]:
    """Map ``fn`` over ``items`` while preserving input order.

    ``backend`` supports: ``serial``, ``thread``, ``process``, ``ray``.
    """

    materialized = list(items)
    if not materialized:
        return []

    workers = _normalize_max_workers(max_workers)
    if workers <= 1 or backend == "serial":
        return _run_serial(materialized, fn)

    if backend == "thread":
        return _run_executor(materialized, fn, max_workers=workers, use_threads=True)
    if backend == "ray":
        return _run_ray(materialized, fn, max_workers=workers)
    return _run_executor(materialized, fn, max_workers=workers, use_threads=False)
