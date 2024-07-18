---
layout: summary
title: "You Need to Pay Better Attention: Rethinking the Mathematics of Attention Mechanism"
giscus_comments: true
bib_id: 2403.01643v2
---

# Overview

The ultimate goal of my personal investigations would be to contribute to the realm of edge models that are personalized in a way a user actually wants. As a result, this paper caught my interest by promising more efficient and performant versions of the Attention Mechanism, as well as being related to the fundamentals of Transformer technology. I decided to read, summarize and re-implement this paper, as further understanding these attention mechanisms could very well prove useful to my own work down the line.

# Three Important Points

## 1.  Motivation for Revisiting Attention

The original “Attention is All You Need” paper introduced the Scaled Dot Product Attention (SDPA) mechanism, but successive work on Transformers has focused mainly on scaling up the approaches outlined in that paper. Furthermore, since increasing scale has been the main focus of most work, we see that models such as GPT4-o or Gemini 1.5 pro have large capabilities, but are prohibitively expensive to train and too large to deploy on edge devices, such as laptops or mobile devices. We also observe that at these large scales, optimizing critical operations will net large gains. Assume a critical function performs 10 matrix multiplications 1 million times. If we can optimize the function to only perform 9 matrix multiplications, then we perform 1 million less matrix multiplications, and therefore use 10% less compute energy and time to run the function! 

The authors observed that vanilla SDPA suffers from performing more matrix multiplications than needed. The authors leverage these facts to reformulate SDPA to cut down on the number of matrix multiplications and parameters used for Optimized and Efficient Attention. For Super Attention, the authors empirically observed that a linear layer between inputs improves the generalizability of the transformer, and combined Efficient Attention with this observation to make a new attention mechanism.

## 2. Optimized, Efficient and Super Attention

### Optimized Attention

Observe that in SDPA, the output of multi head attention will be as follows:

$$
\begin{aligned}
O = \begin{pmatrix}H_1 & H_2 & \dots & H_h \end{pmatrix} W^O \\ 
= \begin{pmatrix}S_1VW_1^V & S_2VW_2^V & \dots & S_hVW_h^V \end{pmatrix} 
   \begin{pmatrix} W_1^O \\ W_2^O \\ \vdots \\ W_h^O\end{pmatrix} \\ 
= S_1VW_1^VW_1^O + S_2VW_2^VW_2^O + \dots + S_hVW_h^VW_h^O
\end{aligned}
$$

This assumes that $W_i^O$ is a matrix with rows $(i-1)d_v + 1, \dots,  i d_v$ of $W^O$ for $i = 1,2,\dots,h$

The authors note that by the rank-nullity theorem, $VW_i^VW_i^O$ will have at most $d_v$ independent columns. This means that instead of performing all these matrix multiplications, we can instead take “slices” of V depending on which head is attending to the sequence and then compute $S_iV_iW_i^O$ instead of $S_iVW_i^VW_i^O$. This clearly cuts down on the amount of matrix multiplications needed as well as the number of parameters stored. 

Slicing V is equivalent to creating $V_i$ that consist of the columns of $(i-1)d_v + 1, \dots, id_v$ of V for $i = 1,2,\dots, h$.

### Efficient Attention

Similarly to Optimized attention, we can observe that for computing the “scores” in SDPA, we compute the following:

$$
\begin{aligned}
Q_i^{'} = Q W_i^Q \text{ and } K_i^{'} = K W_i^{K}\\S_i = \text{softmax}(\frac{Q_i'K_i^{'T}}{\sqrt{d_k}}) \\ = \text{softmax}(\frac{QW_i^Q W_i^K K}{\sqrt{d_k}})
\end{aligned}
$$

The authors establish that any 2 successive linear transformations are redundant, since 2 linear transformations one after another are equivalent to a single linear transformation. Therefore, the authors propose using the same slicing technique on K (the keys) in the attention mechanism to bypass the need for this extra linear transformation. We then see the score computation instead becomes:

$$
\begin{aligned}
Q_i^{'} = Q W_i^Q \\ S_i = \text{softmax}(\frac{Q_i'K_i^{T}}{\sqrt{d_k}})
\end{aligned}
$$

where we now have $K_i$ consisting of the columns of $(i-1)d_v + 1, \dots, id_v$ of K for $i = 1,2,\dots, h$

### Super Attention

Super Attention leverages the gains of efficient attention and an empirical observation the authors made to provide a more effective attention mechanism when it comes to learning for Vision and NLP tasks. The authors observed there is a learnable linear transformation between $Q$ and $K^T$ but not between $K^T$ and $V$. In the original formulation of SDPA, this made sense as the attention scores in $S_i$ correspond to the intuition of “how much attention should be paid” to each of the features of each token of $V_i$.

Adding a linear layer, which they call $W^A \in \mathbb{R}^{\ell \times \ \ell}$ ($\ell$ is the embedding dimension) improves performance, and has the impact of mixing and aligning values token-wise. Furthermore, if $W^A$ is to be used for generative modelling, the restriction that $W^A$ is upper triangular is sufficient to prevent “lookahead” during learning.

We then see that our final formulation for Super Attention is guided by the following equations:

{% include figure.liquid
    path="/assets/img/summaries/super_att.png"
    width="600px"
    class="z-depth-1"
%}

## 3. Method Analysis

### Optimized Attention

Optimized attention reduces the size of an attention layer by $\frac{1}{4}$, and reduces the cost of a single forward pass through the attention layer by $h$ matrix multiplications, where $h$ is the number of heads.  Given the assumption that $d_v = d_k = d_m / h \implies$ there are $d_m^2$ less parameters since standard SDPA has an extra $W_1^V,W_2^V,\dots,W_h^V$ matrix multiplications, each with dimensions $d_md_v$.

### Efficient Attention

Efficient attention follows the same analysis as Optimized Attention. Now, we reduce the size of the attention layer by $\frac{1}{2}$, perform $2h$  less matrix multiplications on a single forward pass and there are  $2d_m^2$ less parameters. 

### Super Attention

The analysis the paper provides for Super Attention is not as rigorous as that of Optimized and Efficient attention, but they observe that it is roughly similar to that of Optimized attention. We retain at least the same gains of reduction in size of an attention layer by $\frac{1}{4}$ and that the cost of a single forward pass will be reduced by $h$ matrix multiplications at minimum. However, we know that the bounds will be slightly better than this, since $W^A \in \mathbb{R}^{\ell \times \ell}$ where $\ell < d_m$ so we will save on some amount of parameters. Since $\ell$ and $d_k$ aren’t necessarily ordered, the analysis lower bounds the gains instead, as detailed above.

# Most Glaring Deficiency

This paper acknowledges this issue, but it is still a notable issue. The authors are resource limited, so they were not able to measure actual performance on training large models, such as LLAMA or OPT. I think that while there are noticeable gains, even on smaller problems such as IMDB sentiment classification and COCO Image Classification, the only way these edits to attention could be made more popular would be through experiments on training these larger models.

# Conclusions For Future Work

This paper has shown me to truly treat everything the way a scientist would: be skeptical of everything. It is quite easy to accept a popular mechanism such as Self Attention as gospel, but these others analyzed it further and discovered optimizations to be made. As a result, this paper helped give me a better eye for finding places where optimizations could be made in ML Systems and Theory.