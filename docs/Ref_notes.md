## How much do Language Models Memorise

### Overview 
In this paper, information theory is applied to understand to what extent language models can memorise information from their training data. The authors focus on Kolmogorov complexity, which measures the smallest representation of a string that can be decoded into the original string on a given computational machine. They use a transformer paired with arithmetic coding as their computational machine, which allows them to estimate the Kolmogorov complexity of the strings in the training data.

Then, with this estimate of Kolmogorov complexity or information content, 