import tensorflow as tf
from tensorflow import keras
from typing import List
class SplineLinear(keras.layers.Layer):

    def __init__(
        self,
        output_dim,
        init_scale=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.output_dim = output_dim
        self.init_scale = init_scale

    def build(self, input_shape):

        self.weight = self.add_weight(
            name="weight",
            shape=(input_shape[-1], self.output_dim),
            initializer=tf.keras.initializers.TruncatedNormal(
                mean=0.0,
                stddev=self.init_scale
            ),
            trainable=True
        )

    def call(self, x):

        return tf.matmul(x, self.weight)


class RadialBasisFunction(keras.layers.Layer):

    def __init__(
        self,
        grid_min=-2.0,
        grid_max=2.0,
        num_grids=8,
        denominator=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_grids = num_grids

        self.grid = tf.linspace(
            grid_min,
            grid_max,
            num_grids
        )

        self.denominator = (
            denominator
            if denominator is not None
            else (grid_max - grid_min) / (num_grids - 1)
        )

    def call(self, x):

        x = tf.expand_dims(x, axis=-1)

        return tf.exp(
            -tf.square(
                (x - self.grid)
                / self.denominator
            )
        )

class Fast_KANLinear(keras.layers.Layer):

    def __init__(
        self,
        output_dim,
        grid_min=-2.0,
        grid_max=2.0,
        num_grids=8,
        use_base_update=True,
        spline_weight_init_scale=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.output_dim = output_dim

        self.num_grids = num_grids

        self.use_base_update = use_base_update

        self.layernorm = keras.layers.LayerNormalization()

        self.rbf = RadialBasisFunction(
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids
        )

        self.spline_weight_init_scale = (
            spline_weight_init_scale
        )

    def build(self, input_shape):

        input_dim = input_shape[-1]

        self.spline_linear = SplineLinear(
            self.output_dim,
            self.spline_weight_init_scale
        )

        self.spline_linear.build(
            (
                None,
                input_dim * self.num_grids
            )
        )

        if self.use_base_update:

            self.base_linear = keras.layers.Dense(
                self.output_dim
            )

    def call(self, x):

        spline_basis = self.rbf(
            self.layernorm(x)
        )

        batch_size = tf.shape(x)[0]

        spline_basis = tf.reshape(
            spline_basis,
            (
                batch_size,
                -1
            )
        )

        ret = self.spline_linear(
            spline_basis
        )

        if self.use_base_update:

            base = self.base_linear(
                tf.nn.silu(x)
            )

            ret = ret + base

        return ret

class FastKAN(keras.Model):

    def __init__(
        self,
        layers_hidden: List[int],
        grid_min=-2.0,
        grid_max=2.0,
        num_grids=8,
        use_base_update=True,
        spline_weight_init_scale=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.layers_kan = []

        for in_dim, out_dim in zip(
            layers_hidden[:-1],
            layers_hidden[1:]
        ):

            self.layers_kan.append(
                Fast_KANLinear(
                    output_dim=out_dim,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    use_base_update=use_base_update,
                    spline_weight_init_scale=spline_weight_init_scale
                )
            )

    def call(self, x):

        for layer in self.layers_kan:

            x = layer(x)

        return x