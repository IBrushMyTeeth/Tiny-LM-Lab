# Width to Depth Ratio in Transformers:

Define the width-to-depth ratio as:

`width_depth_ratio = d_model / num_layers`

Where:

- `d_model` = Transformer hidden dimension
- `num_layers` = number of Transformer blocks

### Why it matters

- **Width enables parallel computation.** Many operations within a layer can execute simultaneously on modern hardware.

- **Depth enables sequential composition.** Each layer transforms the representation produced by previous layers, allowing progressively more complex features and interactions.

- **Depth affects latency.** A forward pass must traverse the layers in sequence, so adding layers increases the sequential computation path.

- **Width and depth provide different capabilities.** A wider model has more representational capacity per layer, while a deeper model permits more successive transformations.

- **Hardware matters.** The best shape depends on accelerator memory, communication overhead, batch size, and whether training or inference is the priority.


### Empirical rule of thumb

Many decoder-only language models have a width-to-depth ratio around 100-160. A value around 128 is a useful reference point, not a universal optimum.

Examples:

- GPT-3 175B: approximately 128
- Mistral 7B: approximately 128
- LLaMA 2 70B: approximately 102
- PaLM 540B: approximately 156

### Practical takeaway

Use `d_model / num_layers` as a descriptive architecture statistic, not a standalone quality metric.

A ratio near 128 is a reasonable initial reference for a conventional decoder-only LLM, but architecture selection should also account for parameter budget, training compute, data, target latency, and task-specific requirements.