import abc
from abc import ABC
from typing import Any, Callable, Optional

from mava.components.jax.component import Component
from mava.core_jax import SystemBuilder
from mava.systems.jax.mamcts.reanalyse_worker import ReanalyseWorker


class ReanalyseCallback(ABC):
    """Abstract base class used to build new components. \
        Subclass this class and override any of the relevant hooks \
        to create a new system component."""

    def on_reanalyse_worker_init_start(self, reanalyse_worker: ReanalyseWorker) -> None:
        """[summary]"""
        pass

    def on_reanalyse_worker_init(self, reanalyse_worker: ReanalyseWorker) -> None:
        """[summary]"""
        pass

    def on_reanalyse_worker_init_end(self, reanalyse_worker: ReanalyseWorker) -> None:
        """[summary]"""
        pass

    def on_reanalyse_worker_utility_fns(
        self, reanalyse_worker: ReanalyseWorker
    ) -> None:
        """[summary]"""
        pass

    def on_reanalyse_worker_step_fn(self, reanalyse_worker: ReanalyseWorker) -> None:
        """[summary]"""
        pass

    def on_reanalyse_worker_step_start(self, reanalyse_worker: ReanalyseWorker) -> None:
        """[summary]"""
        pass

    def on_reanalyse_worker_step(self, reanalyse_worker: ReanalyseWorker) -> None:
        """[summary]"""
        pass

    def on_reanalyse_worker_step_end(self, reanalyse_worker: ReanalyseWorker) -> None:
        """[summary]"""
        pass


class ReanalyseComponent(Component, ReanalyseCallback):
    def on_building_init_start(self, builder: SystemBuilder) -> None:
        if hasattr(builder.store, "reanalyse_components"):
            builder.store.reanalyse_components.append(self)
        else:
            builder.store.reanalyse_components = [self]
