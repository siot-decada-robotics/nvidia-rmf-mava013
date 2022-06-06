from abc import ABC
from types import SimpleNamespace
from typing import List

from mava.callbacks.base import Callback


class ReanalyseWorkerHookMixin(ABC):

    ######################
    # system Reanalyse worker hooks
    ######################

    callbacks: List

    def on_reanalyse_worker_init_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_reanalyse_worker_init_start(self)

    def on_reanalyse_worker_init(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_reanalyse_worker_init(self)

    def on_reanalyse_worker_init_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_reanalyse_worker_init_end(self)

    def on_reanalyse_worker_utility_fns(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_reanalyse_worker_utility_fns(self)

    def on_reanalyse_worker_step_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_reanalyse_worker_step_start(self)

    def on_reanalyse_worker_step(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_reanalyse_worker_step(self)

    def on_reanalyse_worker_step_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_reanalyse_worker_step_end(self)


class ReanalyseWorker(ReanalyseWorkerHookMixin):
    """Reanalyse Worker"""

    def __init__(
        self,
        reanalyse_id,
        trainer_id,
        data_server,
        parameter_server,
        config: SimpleNamespace,
        components: List[Callback] = [],
    ):
        """_summary_

        Args:
            config : _description_
            components : _description_.
        """
        self.id = reanalyse_id
        self.store = config
        self.callbacks = components
        self.store.data_server_client = data_server
        self.store.parameter_server_client = parameter_server
        self.store.trainer_id = trainer_id

        self.on_reanalyse_worker_init_start()

        self.on_reanalyse_worker_utility_fns()

        self.on_reanalyse_worker_init()

        self.on_reanalyse_worker_init_end()

    def step(self) -> None:

        self.on_reanalyse_worker_step_start()

        self.on_reanalyse_worker_step()

        self.on_reanalyse_worker_step_end()

    def run(self) -> None:
        """_summary_"""

        # Run the Reanalyse Worker.
        while True:
            self.step()
