import abc
import jax.numpy as jnp
from typing import Callable
from jax.numpy import ndarray
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split


class IProblem(abc.ABC):
    @abc.abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def get_start_point(self) -> ndarray:
        pass

    @abc.abstractmethod
    def get_func_list(
            self, quantity_of_func: int) -> list[Callable[[ndarray], ndarray]]:
        pass


class MushroomsLogLos(IProblem):
    def __init__(self,
                 file_path: str,
                 test_size: float) -> None:
        assert test_size < 1, "test size >= 1"
        data = load_svmlight_file(file_path)
        self.X, self.Y = data[0].toarray(), data[1]
        self.X_shape = self.X.shape
        for i in range(self.X_shape[0]):
            if (self.Y[i] == 2):
                self.Y[i] = -1.0
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=test_size)

    def get_start_point(self) -> ndarray:
        return jnp.zeros(self.X_shape[1])

    @staticmethod
    def func(w: ndarray, x, y):
        column_q, row_q = x.shape
        return jnp.mean(jnp.log(jnp.ones(column_q) +
                        jnp.exp(-(x @ w.T) * y)).reshape(column_q, 1))

    def get_func_list(
            self, quantity_of_func: int) -> list[Callable[[ndarray], ndarray]]:
        X_split = jnp.array_split(self.X_train, quantity_of_func, axis=0)
        Y_split = jnp.array_split(self.Y_train, quantity_of_func)
        func_list = []
        for i in range(quantity_of_func):
            func_list.append(lambda w, i = i: self.func(w, X_split[i], Y_split[i]))
        return func_list

    def precision_comp(self, w: ndarray) -> ndarray:
        tmp = (self.X_test @ w.T).T * self.Y_test
        if(len(jnp.shape(tmp)) == 1):
            return jnp.count_nonzero(tmp > 0) / len(self.X_test)
        return jnp.count_nonzero(tmp > 0, axis=1) / len(self.X_test)
    