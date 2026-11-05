# MQA VS MHA
### Multi-Head Attention (MHA)

**Fundamental insight**

Instead of having one attention mechanism, we use multiple attention heads in parallel. Each head can learn to focus on different relationships or aspects of the input.

- Each head has its own Query (Q), Key (K), and Value (V) projections.
- Each head independently computes attention.
- The outputs are concatenated and projected back into the model dimension.
- Allows the model to capture different relationships simultaneously.

**The main drawback**

Each head maintains its own Key and Value representations. This increases memory usage, especially during autoregressive inference when the KV cache grows with sequence length.

### Multi-Query Attention (MQA)

**Fundamental insight**

We keep multiple Query heads but share a single Key and Value head across all of them.

- Each attention head still has its own Query projection.
- All heads share the same Keys and Values.
- Reduces the size of the KV cache significantly.
- Reduces memory bandwidth requirements during inference.

**The trade-off**

Sharing Keys and Values reduces the model's flexibility. Different heads can no longer learn completely independent Key and Value representations.

### The key difference

- **MHA:** Multiple Queries, multiple Keys, multiple Values.
- **MQA:** Multiple Queries, one shared Key, one shared Value.

Think of it this way: MHA gives every attention head its own information to work with, whereas MQA makes all heads share the same information while allowing them to ask different questions.

### Why MQA?

During autoregressive generation, previous Keys and Values must be stored in the KV cache.

As sequences grow, this cache becomes increasingly expensive in terms of memory and memory bandwidth.

MQA reduces this cost by sharing Keys and Values across heads, making inference more efficient.

**In short:** MQA trades some attention flexibility for significantly cheaper inference.

# Project Idea:
Compare MHA with GQA (a middle ground between MQA and MHA)