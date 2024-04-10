import numpy as np
import jax
import jax.numpy as jnp
from jax.numpy import ndarray
from tqdm import tqdm
from dataclasses import dataclass
import abc
from typing import Any, Callable

debug_mode = True

######################################################################


class DynamicDataTypeImitation():
    def __init__(self) -> None:
        self.rounding_set = None
        self.bits_per_variable = 4
        self.rounding_set_lengtn = 2**(self.bits_per_variable-1)
        pass

    def _update_rounding_set(self, compressed_data: ndarray) -> None:
        pass #update self.rounding_set

    def dynamic_data_type(self, data: ndarray) -> ndarray:
        pass #conversion to dynamic data type


######################################################################


@dataclass
class AlgoLoggerParams():
    is_logging_node_val: bool = False
    is_logging_master_val: bool = False
    node_x_applied_func: list[Callable[[ndarray], ndarray]] = None
    master_x_applied_func: list[Callable[[ndarray], ndarray]] = None
    logging_rate: float = 0

class AlgoLogger():
    def __init__(self,
                params: AlgoLoggerParams,
                file_name: str,
                ) -> None:
        self.params = params
        self.file_name = file_name
        # use vars(class object instance)
        #как-то дернуть ифну про окружение из под которого запускается, инфу про сам алгоритм и его параметры, что мы хотим собирать
        pass
    def node_logger(self):
        if(self.params.is_logging_node_val):
            pass
        pass
    def master_logger(self):
        if(self.params.is_logging_master_val):
            pass
        pass


######################################################################
class Node():
    def __init__(
            self,
            node_step: Callable[..., None],
            func_in_node: Callable[[ndarray], ndarray],
            compressor: Callable[[ndarray], ndarray],
            x: ndarray) -> None:
        self.step = node_step
        self.func = func_in_node
        self.grad_func = jax.grad(self.func, 0)
        self.compressor = compressor
        self.x = x

    def compute(self) -> ndarray:
        self.step(self)

class Master():
    def __init__(
            self,
            nodes_quantity: int,
            master_step: Callable[..., None],
            x: ndarray) -> None:
        self.nodes_list = []
        self.nodes_quantity = nodes_quantity
        self.step = master_step
        self.x = x

    def compute(self) -> None:
        self.step(self)

class IAlgorithm(abc.ABC):
    def __init__(
            self,
            starting_point: ndarray,
            learning_rate: float,
            iteration_number: int,
            compressor_func: Callable[[ndarray], ndarray],
            nodes_quantity: int,
            node_func: list[Callable[[ndarray], ndarray]]) -> None:
        super().__init__()
        self.startion_point = starting_point
        self.learning_rate = learning_rate
        self.iteration_number = iteration_number
        self.comressor_func = compressor_func
        self.nodes_quantity = nodes_quantity
        self.node_func = node_func

    @abc.abstractmethod
    def _node_step(node) -> None:
        pass

    @abc.abstractmethod
    def _master_step(master) -> None:
        pass

    @abc.abstractmethod
    def _init_set_param_master(self) -> None:
        pass

    @abc.abstractmethod
    def _init_set_param_nodes(self) -> None:
        pass

    @abc.abstractmethod
    def _alg_step(self) -> None:
        pass

    def run_algo(self):
        self._init_set_param_master()
        self._init_set_param_nodes()
        for _ in tqdm(range(self.iteration_number)):
            self._alg_step()
            
#/--------------------------------------------------------EF21------------------------------------------/


  class EF21Node(Node):
    def __init__(self, node_step: Callable[..., None], func_in_node: Callable[[
                 ndarray], ndarray], compressor: Callable[[ndarray], ndarray], x: ndarray) -> None:
        super().__init__(node_step, func_in_node, compressor, x)
        self.c = 0
        self.g = compressor(self.grad_func(self.x)) # node start gradient init


class EF21Master(Master):
    def __init__(self,
                 nodes_quantity: int,
                 master_step: Callable[...,
                                       None],
                 x: ndarray) -> None:
        super().__init__(nodes_quantity, master_step, x)
        self.learning_rate = None


class EF21(IAlgorithm):
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

    def _node_step(self, node: EF21Node):
        node.c = node.compressor(node.grad_func(node.x) - node.g)
        node.g += node.c

    def _master_step(self, master: EF21Master):
        def send_x_to_nodes(nodes_list: list[EF21Node]) -> None:
            for node in nodes_list:
                node.x = master.x

        def all_node_compute(nodes_list: list[EF21Node]):
            for node in nodes_list:
                node.compute()

        def collect_mean_c_from_nodes(nodes_list: list[EF21Node]):
            return jnp.mean(jnp.array([node.c for node in nodes_list]), axis=0)

        master.x -= master.learning_rate * master.g
        send_x_to_nodes(master.nodes_list)
        all_node_compute(master.nodes_list)
        master.g += collect_mean_c_from_nodes(master.nodes_list)

    def _alg_step(self):
        self.master.compute()
        # collect info about alg

    def _init_set_param_master(self) -> None:
        self.master = EF21Master(
            self.nodes_quantity,
            self._master_step,
            self.startion_point)
        self.master.learning_rate = self.learning_rate

    def _init_set_param_nodes(self) -> None:
        def init_nodes() -> list:
            if (debug_mode):
                assert self.nodes_quantity == len(
                    self.node_func), "nodes_quantity != len(node_func))"
            return [
                EF21Node(
                    self._node_step,
                    self.node_func[i],
                    self.comressor_func,
                    self.startion_point) for i in range(
                    self.nodes_quantity)]
        self.master.nodes_list = init_nodes()
        self.master.g = jnp.mean(jnp.array([node.g for node in self.master.nodes_list]), axis = 0) # masters gradient init
        
#/---------------------------------------------------------Marina--------------------------------------------------/

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

#--------------------------------------------------------------Diana------------------------------------------------------------------/

class DianaNode(Node):
    def __init__(self, node_step: Callable[..., None], func_in_node: Callable[[
                 ndarray], ndarray], compressor: Callable[[ndarray], ndarray], x: ndarray, start_rate : float) -> None:
        super().__init__(node_step, func_in_node, compressor, x)
        self.local_grad = 0
        self.local_shifted_grad = 0
        self.local_shift = self.grad_func(x)
        self.rate = start_rate

class DianaMaster(Master):
    def __init__(self,
                 nodes_quantity: int,
                 master_step: Callable[...,
                                       None],
                 x: ndarray) -> None:
        super().__init__(nodes_quantity, master_step, x)
        self.learning_rate = None

class Diana(IAlgorithm) :
    def __init__(self,
                 starting_point: jax.Array,
                 learning_rate: float,
                 iteration_number: int,
                 compressor_func: Callable[...,
                                           Any],
                 nodes_quantity: int,
                 node_func: list[Callable[...,
                                          Any]],
                 diana_learning_rates : list,
                 master_rates) -> None:
        super().__init__(
            starting_point,
            learning_rate,
            iteration_number,
            compressor_func,
            nodes_quantity,
            node_func)
        self.diana_learning_rates = diana_learning_rates # \nu_k \forall i \in range(it_num)
        self.master_rate = master_rates # \alpha_k \forall i \in range(it_num)

    def _node_step(self, node: DianaNode):
        g = node.grad_func(node.x)
        node.local_shifted_grad = node.compressor(g - node.local_shift)
        node.local_shift += node.rate * node.local_shifted_grad

    def _master_step(self, master: DianaMaster):
        def send_info_to_nodes(nodes_list: list[DianaNode]) -> None:
            for node in nodes_list:
                node.x = master.x
                node.rate = master.rates[0]

        def all_node_compute(nodes_list: list[DianaNode]):
            for node in nodes_list:
                node.compute()

        def collect_local_shifted_grad_from_nodes(nodes_list: list[DianaNode]):
            return jnp.mean(jnp.array([node.local_shifted_grad for node in nodes_list]), axis=0)
        
        current_node_rate = master.rates.pop(0)
        current_learning_rate = master.learning_rate.pop(0)
        all_node_compute(master.nodes_list)
        master.g = master.h + collect_local_shifted_grad_from_nodes(master.nodes_list)
        master.x -= current_learning_rate * master.g
        master.h += current_node_rate * (master.g - master.h)
        send_info_to_nodes(master.nodes_list)


    def _alg_step(self):
        self.master.compute()
        # collect info about alg

    def _init_set_param_master(self) -> None:
        self.master = DianaMaster(
            self.nodes_quantity,
            self._master_step,
            self.startion_point)
        start_grad = jax.grad(self.node_func[0], 0)
        self.master.learning_rate = self.diana_learning_rates
        self.master.rates = self.master_rate
        self.master.h = start_grad(self.startion_point)




    def _init_set_param_nodes(self) -> None:
        def init_nodes() -> list:
            if (debug_mode):
                assert self.nodes_quantity == len(
                    self.node_func), "nodes_quantity != len(node_func))"
            return [
                DianaNode(
                    self._node_step,
                    self.node_func[i],
                    self.comressor_func,
                    self.startion_point, 
                    self.master.rates[0]) for i in range(
                    self.nodes_quantity)]
        self.master.nodes_list = init_nodes()
