# from: https://github.com/deepmind/dm-haiku/blob/main/examples/transformer/model.py

"""Transformer model components."""

from typing import Optional

from acme.jax.networks import CategoricalHead
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class CausalSelfAttention(hk.MultiHeadAttention):
    """Self attention with a causal mask applied."""

    def __call__(
        self,
        query: jnp.ndarray,
        key: Optional[jnp.ndarray] = None,
        value: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        key = key if key is not None else query
        value = value if value is not None else query

        # TODO (sasha): remove? Useful check but this is slow
        if query.ndim != 3:
            raise ValueError(
                f"Expect queries of shape [B, T, D]. Got {query.ndim} ({query.shape})"
            )

        seq_len = query.shape[1]
        causal_mask = np.tril(np.ones((1, 1, seq_len, seq_len)))
        mask = mask * causal_mask if mask is not None else causal_mask

        return super().__call__(query, key, value, mask)


class DenseBlock(hk.Module):
    """A 2-layer MLP which widens then narrows the input."""

    def __init__(
        self, init_scale: float, widening_factor: int = 4, name: Optional[str] = None
    ):
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        x = hk.Linear(self._widening_factor * hiddens, w_init=initializer)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(hiddens, w_init=initializer)(x)


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


class EncodeBlock(hk.Module):
    def __init__(self, n_head, n_layers):
        super().__init__(name="decoder")
        self.n_head = n_head
        self.n_layers = n_layers
        self.init_scale = 2.0 / self.n_layers

    def __call__(self, raw_obs, mask=None):
        # attention over actions
        action_attn = CausalSelfAttention(
            num_heads=self.n_head,
            key_size=32,
            model_size=raw_obs.shape[-1],
            w_init_scale=self.init_scale,
            name="decode_action_attention",
        )(raw_obs, mask=mask)
        x = layer_norm(raw_obs + action_attn)

        # mlp
        # TODO is 1 a good widening factor?
        x = layer_norm(x + DenseBlock(self.init_scale, 1, "decode_block_dense")(x))

        return x


class DecodeBlock(hk.Module):
    def __init__(self, n_head, n_layers):
        super().__init__(name="decoder")
        self.n_head = n_head
        self.n_layers = n_layers
        self.init_scale = 2.0 / self.n_layers

    def __call__(self, other_actions, encoded_obs, mask=None):
        # attention over actions
        action_attn = CausalSelfAttention(
            num_heads=self.n_head,
            key_size=32,
            model_size=other_actions.shape[-1],
            w_init_scale=self.init_scale,
            name="decode_action_attention",
        )(other_actions, mask=mask)
        x = layer_norm(other_actions + action_attn)

        # attention over observations and actions
        obs_attn = CausalSelfAttention(
            num_heads=self.n_head,
            key_size=32,  # todo what is key_size?
            model_size=other_actions.shape[-1],  # todo is it: encoded_obs.shape[-1]?
            w_init_scale=self.init_scale,  # todo how to determine this?
            name="decode_obs_attention",
        )(query=encoded_obs, key=x, value=x, mask=mask)
        x = layer_norm(encoded_obs + obs_attn)

        # mlp
        # TODO is 1 a good widening factor? Default is 4, but paper seems to just keep size constant
        x = layer_norm(x + DenseBlock(self.init_scale, 1, "decode_block_dense")(x))

        return x


class Encoder(hk.Module):
    def __init__(self, n_head, n_layers):
        super().__init__()
        self.n_head = n_head
        self.n_layers = n_layers
        self.init_scale = 2.0 / self.n_layers

    def __call__(self, obs):
        # encode obs
        x = layer_norm(DenseBlock(self.init_scale, 1)(obs))

        # pass through attention
        obs_rep = hk.Sequential(
            [EncodeBlock(self.n_head, self.n_layers) for _ in range(self.n_layers)]
        )(x)

        hiddens = obs_rep.shape[-1]
        value_head = hk.Sequential(
            [hk.Linear(hiddens), jax.nn.gelu, layer_norm, hk.Linear(1)]
        )
        value = value_head(obs_rep)

        return value, obs_rep


class Decoder(hk.Module):
    def __init__(self, n_heads, n_blocks, n_actions):
        super().__init__()
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.n_actions = n_actions

    def __call__(self, actions, obs_rep):
        # TODO (sasha): in official implementation this linear is made without bias?
        action_encoder = hk.Sequential(
            [layer_norm, jax.nn.gelu, hk.Linear(obs_rep.shape[-1])]
        )
        encoded_actions = action_encoder(actions)

        decode_blocks = hk.Sequential(
            [DecodeBlock(self.n_heads, self.n_blocks) for _ in range(self.n_blocks)]
        )

        act_obs_attn = decode_blocks(encoded_actions, obs_rep)

        action_head = hk.Sequential(
            [
                hk.Linear(act_obs_attn.shape[-1]),
                jax.nn.gelu,
                layer_norm,
                CategoricalHead(self.n_actions, dtype=jnp.int64),
            ]
        )

        return action_head(act_obs_attn)
