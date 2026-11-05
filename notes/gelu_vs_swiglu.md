
## 1. GELU

- Standard activation function used in Transformer feed-forward networks.
- Uses two linear projections: an expansion and a projection back to the model dimension.
- Simple, efficient, and well-established.


$$  
\operatorname{FFN}(x)=\operatorname{GELU}(xW_1)W_2  
$$

**Trade-off:** Simple architecture with relatively low parameter count, but no explicit multiplicative gating mechanism.

## 2. SwiGLU

- Uses a gated feed-forward network with the SiLU activation function.
- Uses three linear projections instead of two.
- The element-wise product acts as a gate, allowing the network to control which features pass through.
- Commonly used in modern LLMs and often performs better than standard GELU-based FFNs in practice.

$$  
\operatorname{FFN}(x)=  
\left[\operatorname{SiLU}(xW_1)\odot xW_2\right]W_3  
$$

Here, $$\odot$$ denotes element-wise multiplication.

**Trade-off:** Greater representational flexibility and often better empirical performance, at the cost of additional parameters and computation when using the same intermediate dimension.

## 3. Parameter Count

At the same intermediate dimension, SwiGLU uses **50% more parameters** in its feed-forward network, since it requires 3 projections instead of 2. However conventionally the hidden dimension in gated models is slightly reduced (2/3 of non-gated models hidden dimension), to retain an equal parameter count.

