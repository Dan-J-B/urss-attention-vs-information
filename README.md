# urss-attention-vs-information
Project repository for my 2026 URSS project titled 'Do attention patterns in transformers reflect information gain'.


## The research question

This project aims to explore the extent to which, and the conditions under which, transformer architecture's attention weights may reflect information gain. 

## Definition of Information and Information Gain

In this project, we will consider information in the way defined by Claude Shannon as far as possible. As such, in the context of transformers, our notion of **information gain** for a token within a context will be the reduction in Shannon entropy achieved by that token (effectively, the reduction in uncertainty that token provides to the true distribution) - more formally:

$$I(X_t;X_{t-k}|\mathcal{H}\backslash X_{t-k}) = H(X_t|\mathcal{H}\backslash X_{t-k}) - H(X_t|\mathcal{H})$$

Where $X_t$ is the token to be predicted, $X_{t-k}$ is the token for which the information gain is being calculated, and $\mathcal{H}$ is the history (the context) of the sequence up to time $t$. H is the conditional Shannon entropy of the token $X_t$, and I is the mutual information (or information gain) between $X_t$ and $X_{t-k}$ given the rest of the context $\mathcal{H}\backslash X_{t-k}$.

When this definition becomes untenable (or suffers from informational redundancy issues as remarked below), we will have to consider approximations or proxies for information gain. As motivated below, we will consider the following proxies for information/information gain:

1) **Surprisal**: Surprisal is a measure of how unexpected a token is given the context. It is defined as the negative logarithm of the probability of the token given the context, which will be according to a model's learned distribution in this case. Shannon entropy is the expected value of surprisal, so we can expect that surprisal will be a good proxy for information content as long as the model's learned distribution is a good approximation of the true distribution.

2) **Positive Pointwise Mutual Information (PPMI)**: PPMI is a measure of association between two events, in this case, two tokens. It is defined as follows:
$$ PPMI(X_t;X_{t-k}) = \max\left(0, \log\frac{P(X_t,X_{t-k})}{P(X_t)P(X_{t-k})}\right) $$

Where $P(X_t,X_{t-k})$ is the joint probability of the two tokens, approximated via their co-occurrence in the dataset, and $P(X_t)$ and $P(X_{t-k})$ are the marginal probabilities of the two tokens, also approximated via their occurrence in the dataset. PPMI is a measure of how much more likely the two tokens are to occur together than would be expected if they were independent, and can be considered a proxy for mutual information between the two tokens.

3) **KL-Divergence (Mutual Information)**: KL-Divergence is a measure of how one probability distribution diverges from a second (or provides information gain over another) probability distribution. Mutual information arises as a special case of KL-Divergence (specifically, the KL-Divergence between the joint distribution and the product of the marginals, equivalent to comparing distributions under the assumption of independence or not). Then, we can use KL-Divergence in both the sense of mutual information (where we rely on the internal learned distributions of the model to approximate mutual information) and, potentially as an interesting extension, in the sense of determining the information gain from one layer to the next, comparing the learned distributions of the model at different layers.

### Remarks on the definition of information gain

1) This definition requires that the true conditional distributions of the token $X_t$ are known - this will only be true in some synthetic datasets that we will construct. In other cases, we will have to rely on a proxy for either information or the true conditional distributions.

2) There is also an issue of redundant information - consider a toy example where we are predicting the next value in a Fibonacci sequence. By our above definition, we may conclude that the value $X_{t-2}$ provides no information gain because $H(X_t|\mathcal{H}\backslash X_{t-2}) = H(X_t|\mathcal{H})=0$, as long as we have $X_{t-1}$ and $X_{t-3}$ in the context. A similar argument can be made for $X_{t-1}$ as well. However, it is clear that both $X_{t-1}$ and $X_{t-2}$ are informative in some sense, as they define the next term in the sequence. This is certainly a limitation of our definition of information gain, and we will explore how transformers navigate this issue of redundant information.

## Project structure

### Stage 1: Synthetic datasets

As motivated above, we will first investigate the relationship between attention and information gain in scenarios where true conditional distributions are known, such that we can calculate information gain exactly or, at least, have some basis of understanding of the true information gain. 
Therefore, the first stage of this project will be to construct synthetic datasets where the data generating mechanism is known and provides a clear notion of information gain. To this end, we may consider sequences generated by simple mathematical functions, such as Fibonacci sequences, or Markov chains with known transition probabilities. 
Then, the goal will be to train transformer models on these synthetic datasets and to analyse the attention patterns between tokens in relation to the information gain between those tokens, both during and after training.

All files directly pertaining to this stage of the project will contain 'synthetic' in their name.

### Stage 2: NLP and real-world datasets

Following the synthetic datasets, we will move to real world datasets, in the quintessential domain of the transformer - natural language processing. This introduces two key challenges: first, the true conditional distributions are not known, and second, the notion of information gain is less clear. Therefore, we will have to rely on proxies for information gain, which fall into two categories:
1) Reliance on a proxy for the true conditional distributions, such as the learned conditional distributions of a large language model (metrics like surprisal or KL-divergence could be used to estimate information gain).
2) Reliance on a proxy for information itself, which might look like PPMI (Positive Pointwise Mutual Information), that acts as a proxy for mutual information between tokens, independent of any specific conditions (context).

Then, we will follow one of two implementations:
1) Train transformer models on real-world datasets, analysing the attention patterns in relation to the information gain proxies both during and after training.
2) Use pre-trained, open source transformer models and analyse the attention patterns in much the same way as in the first implementation, though without the training analysis (unless some fine-tuning or post-training is performed).

