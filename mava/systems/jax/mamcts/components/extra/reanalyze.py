@dataclass
class ReanalyseWorkerConfig:
    pass


class ReanalyseWorker(Component):
    def __init__(
        self,
        config: ReanalyseWorkerConfig = ReanalyseWorkerConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config
        self.dataset_iterator = None
        self.data_server_client = None

    def on_building_trainer_dataset(self, builder: SystemBuilder) -> None:

        self.dataset_iterator = builder.store.dataset_iterator
        self.data_server_client = builder.store.data_server_client

    def reanalyse_data(self):
        pass

    def on_parameter_server_run_loop(self, server: SystemParameterServer) -> None:
        if self.dataset_iterator and self.data_server_client:
            self.reanalyse_data()

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "reanalyse_worker"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """
        Optional class which specifies the
        dataclass/config object for the component.
        """
        return ReanalyseWorkerConfig
