

The MLP expansion ratio is defined as:

`expansion_ratio = d_ff / d_model`

Where:

- `d_model` = Transformer hidden dimension
- `d_ff` = MLP intermediate dimension

| Model        | Expansion ratio | Description                 | Notes                                                   |
| ------------ | --------------- | --------------------------- | ------------------------------------------------------- |
| PaLM         | 4.00            | Dense Transformer LLM       | Uses a gated MLP (SwiGLU).                              |
| Mistral 7B   | 3.50            | Decoder-only LLM            | Uses a SwiGLU MLP.                                      |
| LLaMA-2 70B  | 3.50            | Decoder-only LLM            | Intermediate dimension: 28,672; model dimension: 8,192. |
| LLaMA 70B    | 2.68            | Decoder-only LLM            | Verify the exact model variant.                         |
| Qwen 14B     | 2.67            | Decoder-only LLM            | Approximately 8/3.                                      |
| DeepSeek 67B | 2.68            | Decoder-only LLM            | Verify the exact model variant.                         |
| Yi 34B       | 2.85            | Decoder-only LLM            | Uses a gated MLP.                                       |
| T5 v1.1      | 2.50            | Encoder-decoder Transformer | Uses a GEGLU feed-forward network.                      |

### Important Note

Although the default choices of 4 for non-gated and 2,66 for gated models have worked well for nearly all models in LLMs, this ratio is just another hyper parameter.   
There is therefore no rule constraining us to such values and Google's T5 with a ratio of 64 is the perfect example.

### Conclusion
- Standard MLPs often use an expansion ratio of 4.
- Gated MLPs such as SwiGLU and GEGLU often use a smaller intermediate dimension.
- A ratio of 8/3 (approximately 2.67) is common for gated MLPs.
- The ratio alone does not fully describe the MLP parameter count: gated MLPs typically use three projection matrices instead of two.
- Exact ratios can vary by model version and configuration.