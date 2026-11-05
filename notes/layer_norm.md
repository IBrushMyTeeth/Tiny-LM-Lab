
## LayerNorm

Layer Normalization (LayerNorm) is a deep learning technique that standardizes the activations of a neural network by computing the mean and variance across the **feature dimensions of each individual sample**, rather than across the batch

Here, activations simply means the numerical values flowing through a neural network. They don't have to come from a ReLU or GELU.

For example, in a Transformer, a token's hidden representation might be:
$X = [2, 4, 6, 8]$

LayerNorm normalizes these values across the feature dimension. It can operate on:
- Outputs of linear layers.
- Token embeddings and residual-stream representations.
- Outputs of attention or MLP blocks.
- Outputs of nonlinearities such as ReLU or GELU.

## Why?

- Activation stability: Prevents values from the extremes
- Optimization stability: Subsequent layers receive inputs with more controlled statistics, which can make learning easier.
- Gradient flow: In pre-norm Transformers, normalization placement helps preserve a more direct gradient path through residual connections.
- Not a complete solution: Exploding gradients can still occur due to the network's weights, depth, attention dynamics, or learning rate. Gradient clipping is a more direct intervention for excessive gradient norms.

## How

Part 1:
For each single sample, go through its features: x1, x2, x3, ...
Calculate the mean and variance across this samples features:


$$
\mu = \frac{1}{X} \sum_{i=1}^X x_i
$$
$$
\sigma² = \frac{1}{X}\sum_{i=1}^X (x_i - \mu)²
$$
Part 2:
**Normalize**: Subtract the mean and divide by the square root of the variance (plus a small epsilon ε for numerical stability)

$$
x_i' = \frac{x_i - \mu}{\sqrt{\sigma² + \epsilon}}
$$
part 3:
**Scale and Shift (Affine Transformation)**: Apply learnable parameters—weight (γ) and bias (β)—to allow the network to restore representation power:

$$
x_i'' = \gamma_i \cdot x_i'+\beta_i  
$$

Where
- Scale $\gamma_i$:  Controls how much each normalized feature is amplified or reduced.
- Shift $\beta_i$:  Controls the offset of each feature, moving it up or down.
## Why do we need them?

Normalization forces the features to have zero mean and unit variance (approximately, before the scale and shift). This is useful for stability, but it could constrain the representations unnecessarily.

The learnable parameters give the model flexibility to recover a different scale and offset if that helps the task.