from mava.systems.idrqn.components.building import DRQNExtrasSpec


class QmixExtrasSpec(DRQNExtrasSpec):
    def get_network(self, builder):
        return builder.store.network_factory()[0]
