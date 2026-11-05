# Normalization Architectures

## 1. Pre-Norm (LayerNorm)

- Normalizes the input before each Transformer sublayer.
- Improves training stability and gradient flow, particularly in deeper Transformers.
- Uses mean-centering and variance normalization.
- Includes learnable scaling and shifting parameters $\gamma$ and  $\beta$.
- Generally provides strong normalization and representational flexibility, at the cost of additional computation.


## 2. Pre-Norm (RMSNorm)

- Normalizes the input using its root mean square, without mean-centering.
- Uses fewer operations and can reduce memory traffic and computational overhead.
- Typically uses only a learnable scaling parameter $\gamma$, without an additive shift $\beta$.
- Often achieves performance close to LayerNorm while being computationally cheaper.
- Widely adopted in modern LLM architectures.


**Trade-off:** Slightly less normalization flexibility in exchange for efficiency. RMSNorm is not universally as effective as LayerNorm, but is often nearly as effective in practice.

## 3. Double-Norm

- Applies normalization both before and after a sublayer (e.g. attention or MLP).
- Can improve training stability and control the scale of intermediate representations.
- Introduces additional normalization parameters and computational overhead.
- May be beneficial in deeper architectures or when training stability is a concern.

**Trade-off:** Potentially improved stability at the cost of additional parameters and computation. Benefits are architecture- and training-dependent.