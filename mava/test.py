import jax
import jax.numpy as jnp
from functools import partial


class Super:
    def __init__(self, x):
        self.x = x

    @classmethod
    @partial(jax.jit, static_argnames="cls")
    def inner(cls, y):
        return y + 1

    def inner_caller(self, x):
        print(self.inner(x))


class Sub(Super):
    @classmethod
    @partial(jax.jit, static_argnames="cls")
    def inner(cls, y):
        return y - 1


s = Super(2)
s.inner_caller(1)
