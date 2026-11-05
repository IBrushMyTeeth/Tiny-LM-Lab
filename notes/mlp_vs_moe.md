# Mixture of Experts (MoE)

## High-level idea

A dense MLP uses all its parameters for every token. However, a key insight is that a neural network does not necessarily need to use every parameter for each input.

Instead, we can selectively activate a small subset of parameters depending on the input token.

This allows us to increase the total number of parameters in a model without proportionally increasing the number of active parameters (computation) per token.

## Dense MLP

Suppose we have an MLP with:

- Input dimension: $d = 1024$
- Hidden dimension: $h = 4096$
- Output dimension: $d = 1024$

This MLP consists of two transformations:

1. **Expansion:** $d \rightarrow h$
2. **Projection:** $h \rightarrow d$

The total number of parameters is therefore:

$$  
P = dh + hd = 2dh  
$$

$$  
P = 2 \cdot 1024 \cdot 4096 = 8.39M  
$$

Every parameter in this MLP is used for every input token.

## Mixture of Experts (MoE)

Instead of having one MLP, imagine having 8 independent MLPs, called **experts**.

Each expert has exactly the same architecture as our original MLP.

Therefore, the total number of parameters becomes:

$$  
P_{\text{total}} = 8 \cdot 8.39M = 67.1M  
$$

We now have 8 times as many parameters as in the original dense MLP.

However, we introduce a **router** that decides which experts should process each input token.

### Router and Top-K selection

The router determines which experts are activated for a given token.

For example, suppose we have 8 experts and use Top-2 routing:

- Token 1 → Experts 2 and 5
- Token 2 → Experts 1 and 7
- Token 3 → Experts 3 and 4

Each token is only processed by 2 out of the 8 experts.

The router typically calculates a score for each expert and selects the top K experts with the highest scores.

### Active vs. total parameters

Although the model contains 8 experts, only 2 experts are activated for each token.

Each expert contains 8.39M parameters.

Therefore, the number of active expert parameters per token is:

$$  
P_{\text{active}} = 2 \cdot 8.39M = 16.78M  
$$

We can distinguish between:

- **Total parameters:** All parameters across all experts, regardless of whether they are activated.
- **Active parameters:** Parameters belonging to the experts selected by the router for a particular token.

### Important: What if we want the same number of active parameters?

In the example above, we increased the active parameter count by a factor of 2 compared to the original dense MLP.

But what if we want to maintain approximately the same active parameter count?

We can simply make each expert smaller.

For example, we can reduce the hidden dimension of each expert from 4096 to 2048.

Each expert now has:

$$  
P_{\text{expert}} = 2 \cdot 1024 \cdot 2048  
$$

$$  
P_{\text{expert}} = 4.19M  
$$

With 8 experts, the total number of parameters becomes:

$$  
P_{\text{total}} = 8 \cdot 4.19M = 33.55M  
$$

With Top-2 routing, the active parameter count per token is:

$$  
P_{\text{active}} = 2 \cdot 4.19M = 8.39M  
$$

Compare this to the original dense MLP:

|Metric|Dense MLP|MoE|
|---|---|---|
|Number of experts|1|8|
|Hidden dimension|4096|2048|
|Total parameters|8.39M|33.55M|
|Active parameters/token|8.39M|8.39M|

We have now achieved:

- 4 times as many total expert parameters.
- Approximately the same number of active expert parameters per token.

The MoE can therefore have a much larger parameter capacity while maintaining approximately the same active parameter count.

## Shared Experts

### The problem

In a standard MoE, all experts are independently selected by the router.

However, some knowledge and representations are useful for almost every token.

For example, in NLP:

- Basic grammatical structures.
- Common language patterns.
- General semantic relationships.

Without shared experts, multiple routed experts might independently learn some of these common patterns.

This can lead to redundancy across experts.

### The solution: Shared experts

Introduce one or more experts that are **always activated**, regardless of the router's decision.

We now have two types of experts:

1. **Shared experts:** Always activated for every token.
2. **Routed experts:** Dynamically selected by the router.

For example, suppose we have:

- 1 shared expert.
- 8 routed experts.
- Top-2 routing.

Every token is processed by:

- The shared expert.
- Two selected routed experts.

The output becomes:

$$  
y = E_{\text{shared}}(x) +  
\sum_{i \in \operatorname{TopK}(x)}  
g_i(x) E_i(x)  
$$

Where:

- $E_{\text{shared}}(x)$ is the shared expert's output.
- $E_i(x)$ is the output of routed expert $i$.
- $g_i(x)$ is the routing weight assigned to expert $i$.

The shared expert's output is combined with the weighted outputs of the routed experts.

### Why is this useful?

The shared expert provides a common foundation of knowledge that is available to every token.

This potentially allows the routed experts to focus on more specialized patterns rather than repeatedly learning the same general knowledge.

An important distinction: Shared experts are not frozen or manually programmed. They are trained through backpropagation just like the routed experts.

This approach is known as **Shared Expert Isolation**, introduced as part of DeepSeekMoE.

## Fine-Grained Experts

### The problem

In a standard MoE, experts can be relatively large.

For example:

- 8 experts.
- Hidden dimension of 4096 per expert.
- Top-2 routing.

Each token activates two relatively large experts.

The problem is that selecting only two large experts gives the router relatively limited flexibility in how it combines expert knowledge.

### The solution: Fine-grained expert segmentation

Instead of having a few large experts, we split them into many smaller experts.

For example, imagine splitting each of our original 8 experts into 4 smaller experts.

We now have:

- 32 experts instead of 8.
- Hidden dimension of 1024 instead of 4096.
- Top-8 routing instead of Top-2.
### Parameter comparison

**Standard MoE:**

8 experts, each with hidden dimension 4096.

$$  
P_{\text{total}} = 8 \cdot 2 \cdot 1024 \cdot 4096  
$$

$$  
P_{\text{total}} = 67.1M  
$$

Active parameters:

$$  
P_{\text{active}} = 2 \cdot 2 \cdot 1024 \cdot 4096  
$$

$$  
P_{\text{active}} = 16.78M  
$$

**Fine-grained MoE:**

32 experts, each with hidden dimension 1024.

$$  
P_{\text{total}} = 32 \cdot 2 \cdot 1024 \cdot 1024  
$$

$$  
P_{\text{total}} = 67.1M  
$$

Active parameters:

$$  
P_{\text{active}} = 8 \cdot 2 \cdot 1024 \cdot 1024  
$$

$$  
P_{\text{active}} = 16.78M  
$$

We have maintained exactly the same total and active expert parameter counts.

### Why is this useful?

The important benefit is combinatorial flexibility.

With 8 experts and Top-2 routing, the number of possible expert combinations is:

$$  
\binom{8}{2} = 28  
$$

With 32 experts and Top-8 routing:

$$  
\binom{32}{8} = 10,518,300  
$$

This is a huge increase in the number of possible combinations!

The router now has many more ways to combine different specialized experts.

Instead of selecting two relatively large experts, it can select eight smaller experts that collectively provide a more targeted combination of knowledge.

This is the central motivation behind **Fine-Grained Expert Segmentation** in DeepSeekMoE.

Note that more possible combinations do not automatically mean better performance. The experts must actually learn useful complementary representations, and the router must learn to select them effectively.

## Combining Both: DeepSeekMoE

DeepSeekMoE combines the two concepts:

1. **Shared Expert Isolation:** Some experts are always activated to capture common knowledge.
2. **Fine-Grained Expert Segmentation:** More numerous, smaller experts provide greater flexibility in routing and specialization.

The architecture therefore contains:

- Shared experts that process every token.
- Many smaller routed experts.
- A router that selects a subset of the routed experts for each token.
    

The final output is the combination of the shared expert outputs and the weighted routed expert outputs.

The overall objective is to reduce redundancy while allowing more flexible expert specialization.

## True Cost vs. Reward: The Big Question

The parameter counts above exclude the router, which introduces additional parameters.

The router overhead is often relatively small compared with the expert parameters, although it can become significant for very small models.

Furthermore, routing introduces computational overhead:

- Calculating routing scores.
- Selecting experts.
- Dispatching tokens.
- Combining expert outputs.

Therefore, it is incorrect to assume that equal active parameter counts guarantee equal computational costs.

The architecture trades additional routing complexity and potentially some computational overhead for more total representational capacity.

MoE is an attractive architecture because research has demonstrated that sparse MoE models can achieve better language modeling quality than dense models under certain compute and parameter budgets.

However, the big question for our project is:

**Given our limited resources (CPU-only compute), is it worth paying this overhead to obtain more representational power?**

This is an empirical question that we should investigate rather than assume.

## Project experiment idea with for example model_d = 256:

Dense baseline

- FFN dimension 1024
- GELU activation

MoE alternative

- 1 shared expert
- 7 routed experts
- Expert hidden dimension 256
- Top-3 routing
- GELU activation
- Load-balancing auxiliary loss