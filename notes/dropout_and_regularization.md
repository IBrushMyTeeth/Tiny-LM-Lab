# Dropout and Regularization

### Fundemental insight

We almost never lack data. The internet is a rich and almost infinite source of data. And sure, we can be picky and filter much of it out, but even so most LLMs have a huge abundance of data (trillions of tokens) meaning that even getting through one single epoch is unexpected. We therefore never see the same sample twice.

### Arguments against Regularization
* This abundance of data means that we only do a single epoch
* Overfitting is a very unlikely result from a single pass
* SGD cannot effectively memorize the corpus with only a single pass
* Dropout adds computational overhead

### These are very reasonable arguments but what is done in practice?

| Model                | Dropout | Weight decay                        |
| -------------------- | ------- | ----------------------------------- |
| Original Transformer | 0.1     | Not specified as a standard setting |
| GPT-2                | 0.1     | 0.1                                 |
| T5                   | 0.1     | Configuration-dependent             |
| GPT-3                | 0.0     | 0.1                                 |
| LLaMA                | 0.0     | 0.1                                 |

Note: Newer models don't use dropout but add weight decay

### Why weight decay?

Weight decay is often introduced as a regularizer that discourages excessively large weights. However, its role in large-scale training is not limited to preventing overfitting.

With AdamW, weight decay directly shrinks eligible parameters during optimization, separately from the gradient-based update. This changes the optimization trajectory and can improve the final solution even when classical overfitting is not the main concern.