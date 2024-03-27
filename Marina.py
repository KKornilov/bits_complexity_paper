class MarinaNode(Node):
    def __init__(self, node_step: Callable[..., None], func_in_node: Callable[[
                 ndarray], ndarray], compressor: Callable[[ndarray], ndarray], x: ndarray) -> None:
        super().__init__(node_step, func_in_node, compressor, x)
        self.probability = 0


class MarinaMaster(Master):
    def __init__(self,
                 nodes_quantity: int,
                 master_step: Callable[...,
                                       None],
                 x: ndarray) -> None:
        super().__init__(nodes_quantity, master_step, x)
        self.learning_rate = None
        self.g = 0
        self.probability = 0

class Marina(IAlgorithm) :
    def __init__(self,
                 starting_point: jax.Array,
                 learning_rate: float,
                 iteration_number: int,
                 compressor_func: Callable[...,
                                           Any],
                 nodes_quantity: int,
                 node_func: list[Callable[...,
                                          Any]]) -> None:
        super().__init__(
            starting_point,
            learning_rate,
            iteration_number,
            compressor_func,
            nodes_quantity,
            node_func)

    def _node_step(self, node: MarinaNode):
        node.x -= self.master.learning_rate * node.g
        node.g = node.grad_func(node.x) if node.probability == 1 else node.g + node.compressor(node.grad_func(node.x)
         - node.grad_func(self.master.learning_rate * node.g + node.x))

    def _master_step(self, master: MarinaMaster):
        def send_grad_to_nodes(nodes_list: list[MarinaNode]) -> None:
            for node in nodes_list:
                node.g = master.g
                node.probability = self.master.probability

        def all_node_compute(nodes_list: list[MarinaNode]):
            for node in nodes_list:
                node.compute()

        def collect_grad_from_nodes(nodes_list: list[MarinaNode]):
            return jnp.mean(jnp.array([node.g for node in nodes_list]), axis=0)

        self.master.probability = np.random.choice([1, 0], p = [0.1, 0.9])
        send_grad_to_nodes(master.nodes_list)
        all_node_compute(master.nodes_list)
        master.g = collect_grad_from_nodes(master.nodes_list)

    def _alg_step(self):
        self.master.compute()
        # collect info about alg

    def _init_set_param_master(self) -> None:
        self.master = MarinaMaster(
            self.nodes_quantity,
            self._master_step,
            self.startion_point)
        start_grad = jax.gradient(self.node_func[0], 0)
        self.master.learning_rate = self.learning_rate
        self.master.g = start_grad(self.startion_point)


    def _init_set_param_nodes(self) -> None:
        def init_nodes() -> list:
            if (debug_mode):
                assert self.nodes_quantity == len(
                    self.node_func), "nodes_quantity != len(node_func))"
            return [
                MarinaNode(
                    self._node_step,
                    self.node_func[i],
                    self.comressor_func,
                    self.startion_point) for i in range(
                    self.nodes_quantity)]

        self.master.nodes_list = init_nodes()
