# Descriptions of the quantization parameters

- `default_integer_bits`: Number of integer bits used for quantization.
- `default_fractional_bits`: Number of fractional bits used for quantization. For ReLU, one extra bit is added to the default value when adding compression layers, since ReLU is unsigned and requires no sign bit.
- `enable_quantization`: Enables quantization.
- `hgq_gamma`: Scales the HGQ loss. If set too high, can cause the entire model to be pruned.
- `hgq_heterogeneous`: If `true`, HGQ learns a separate set of bits for each weight in the model. If `false`, learns one set of bits per layer.
- `layer_specific`: Layers that use non-default quantization bits. A default config containing all layers can be generated using `pquant.add_default_layer_quantization_pruning_to_config`.
- `use_high_granularity_quantization`: If `true`, uses HGQ instead of fixed quantizers.
- `use_real_tanh`: If `true`, applies the real tanh function before quantization. If `false`, uses hard tanh.
- `use_relu_multiplier`: If `true`, multiplies the input of `QuantizedReLU` by a learned multiplier before the operation: $\text{inputs} = \text{inputs} \times 2^{\text{round}(m)}$, where $m$ is the learned multiplier. $m$ is initialized to $-1$, so at the start of training inputs are scaled by $0.5$.
-  `use_symmetric_quantization`: If `true`, enforces `minimum_quantized_value` == `-maximum_quantized_value`.
