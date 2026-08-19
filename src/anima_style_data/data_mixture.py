"""Deterministic whole-microbatch dataset mixing."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from fractions import Fraction
from typing import Any, Iterator


def auxiliary_step(step: int, fraction: float) -> bool:
    """Return a low-discrepancy Bernoulli schedule without random bursts."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("mixture fraction must be in [0,1]")
    if fraction == 0.0:
        return False
    if fraction == 1.0:
        return True
    step = int(step)
    if step < 0:
        raise ValueError("step must be non-negative")
    ratio = Fraction(str(fraction)).limit_denominator(10_000)
    return (
        ((step + 1) * ratio.numerator) // ratio.denominator
        != (step * ratio.numerator) // ratio.denominator
    )


class ConstantRatioBatchMixer:
    """Route complete microbatches while retaining each loader's curriculum."""

    def __init__(
        self,
        primary: Any,
        auxiliary: Any,
        *,
        auxiliary_fraction: float,
        primary_name: str = "anima",
        auxiliary_name: str = "megastyle",
    ) -> None:
        if primary.batch_size != auxiliary.batch_size:
            raise ValueError("mixed loaders must use the same batch size")
        if not 0.0 < auxiliary_fraction < 1.0:
            raise ValueError("enabled mixture needs a fraction strictly between 0 and 1")
        self.primary = primary
        self.auxiliary = auxiliary
        self.auxiliary_fraction = float(auxiliary_fraction)
        self.primary_name = str(primary_name)
        self.auxiliary_name = str(auxiliary_name)
        self.batch_size = int(primary.batch_size)

    @property
    def loaders(self) -> tuple[Any, Any]:
        return self.primary, self.auxiliary

    def domain_for_step(self, step: int) -> str:
        return (
            self.auxiliary_name
            if auxiliary_step(step, self.auxiliary_fraction)
            else self.primary_name
        )

    def load_step(self, step: int) -> dict[str, Any]:
        use_auxiliary = auxiliary_step(step, self.auxiliary_fraction)
        loader = self.auxiliary if use_auxiliary else self.primary
        batch = loader.load_step(step)
        batch["data_domain"] = (
            self.auxiliary_name if use_auxiliary else self.primary_name
        )
        return batch

    def prefetch(
        self, start_step: int, steps: int, workers: int = 1, depth: int = 4
    ) -> Iterator[dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures: dict[int, Future[dict[str, Any]]] = {}
            next_step = int(start_step)
            stop = next_step + int(steps)
            for step in range(int(start_step), stop):
                while next_step < stop and len(futures) < max(1, depth):
                    futures[next_step] = executor.submit(self.load_step, next_step)
                    next_step += 1
                yield futures.pop(step).result()
