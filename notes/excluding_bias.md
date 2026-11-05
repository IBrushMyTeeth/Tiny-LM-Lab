Many modern language models, particularly Transformer-based architectures, omit explicit bias terms from some or all of their linear layers because biases often provide relatively little benefit compared with their computational, memory, and implementation costs.

## What does bias do?
A standard linear transformation is:
$$
y = xW + b
$$
Where  
x: is the input  
W: is the weight matrix  
b: is the bias vector  

The bias allows the model to shif the input independently of the input. 

Without the bias the input 0 is constrained to 0:
$$
f(0) = 0
$$


## Why do modern transformers remove bias?

Model capacity arguments:
* LayerNorm already provides an offset
* Biases are some times mathematically redundent: e.g.  2 consecutive linear layers
* Bias can also be removed from self attention (In fact the key bias is completely redundant)

Computational arguments:
- The number of trainable parameters.
- Memory requirements for model weights and optimizer states.
- A small amount of computation.
- Some implementation complexity.
* Some study also shows that dropping the bias stabilizes training!


## Key lesson
We can try to remove bias as long as we keep a minimal meaningful representational capacity and not completely constrain ourselves to scaling operations