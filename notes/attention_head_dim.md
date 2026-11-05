# Attention Head Dimension

The attention width ratio is defined as:

`attention_width_ratio = num_heads * head_dim / model_dim`

Where:

- `num_heads` = number of query attention heads
- `head_dim` = dimension of each query head
- `model_dim` = Transformer residual-stream hidden dimension

| Model            | Num heads  | Head dim   | Model dim  | Heads * head dim / model dim |
| ---------------- | ---------- | ---------- | ---------- | ---------------------------- |
| Transformer Base | 8          | 64         | 512        | 1.00                         |
| GPT-2 Small      | 12         | 64         | 768        | 1.00                         |
| GPT-3 175B       | 96         | 128        | 12,288     | 1.00                         |
| PaLM             | 48         | 258        | 18,432     | 1.48                         |
| LLaMA 2 70B      | 64         | 128        | 8,192      | 1.00                         |
| Mistral 7B       | 32         | 128        | 4,096      | 1.00                         |
| Qwen 14B         | 40         | 128        | 5,120      | 1.00                         |
| DeepSeek 67B     | 64         | 128        | 8,192      | 1.00                         |
| LaMDA            | 128        | 128        | 8192       | 2.00                         |
| T5               | 128        | 128        | 1024       | 16.00                        |
| GPT-4            | Not public | Not public | Not public | N/A                          |

### Notes and common design patterns

- **The ratio is usually 1.0.** The concatenated query heads have a total dimension equal to the model dimension.
- **Head dimension is often 64 or 128.** Many modern LLMs keep the head dimension fixed and scale the number of heads with the model dimension.
- **GQA reduces KV heads, not query heads.** For example, Mistral 7B uses 32 query heads but only 8 key/value heads. The ratio above uses 32.
- **MHA vs. GQA:** In multi-head attention (MHA), each query head has its own key/value heads. In grouped-query attention (GQA), multiple query heads share key/value heads, reducing KV-cache memory.
- **Exceptions exist.** Some architectures use different query and output widths, latent attention, or other attention designs. In those cases, the ratio may differ from 1.0 or may not be directly comparable.
- **GPT-4 dimensions are not publicly specified.** Avoid inferring its head count or dimensions from the model's parameter count.

### Main takeaway

For conventional Transformer attention, the common design is:

`num_heads * head_dim = model_dim`

This is a useful baseline when comparing architectures. However, it describes the query projection width, not the KV-cache size or the entire attention mechanism.