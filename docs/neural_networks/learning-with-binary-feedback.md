---
title: Learning with Binary Feedback
description: How can we train a model when the only supervision is a binary right/wrong signal? A maximum-likelihood treatment for classification and CTC.
tags:
  - neural
  - networks
  - feedback
  - learning
  - ctc
  - reinforcement
---

# Learning with Binary Feedback

Assume we do not have access to the ground truth of a dataset. But we have a system which tells us whether a prediction was correct or incorrect. Can we learn using just this feedback?

As far as systems that use maximum likelihood estimation go, we believe we can.

## How do we learn?

Let's start with formulating a system.
We have a network with weights $\mathcal{N}_w$ and an input $\mathbf{x}$.
Now let's say the network makes a prediction and based on that we deduce that the output should be $l$. We feed this $l$ into our feedback system and we get a feedback signal $f$, 0 or 1.
We want to update the weights using the feedback.

### Case 1: The feedback is positive

If the model predicts the correct output, we want the model to be more confident about it and we can use this output as ground truth. In this case we maximize the likelihood, $p(l \mid \mathbf{x})$.

Or we can minimize the negative log-likelihood of the probability.

$$
-\ln(p(l \mid \mathbf{x}))
$$

### Case 2: The feedback is negative

If the model predicts the wrong output, we want the model to be less confident about it and so we want to maximize the likelihood, $p(l^c \mid \mathbf{x}) = 1 - p(l \mid \mathbf{x})$.

Or we can minimize the negative log-likelihood of the probability.

$$
-\ln(1 - p(l \mid \mathbf{x}))
$$

Before starting with concrete types, softmax is a very common activation function used in these models. It is defined as: for input vector $(u_1, u_2, \dots, u_n)$ the softmax is another vector of same length $(y_1, y_2, \dots, y_n)$, such that

$$
\begin{equation}
y_i = \frac{e^{u_i}}{\sum_{j=1}^n e^{u_j}}
\end{equation}
$$

The derivative of this is very common and will be seen multiple times throughout.

$$
\begin{equation}
\frac{\partial y_i}{\partial u_j} = y_i(\delta_{ij} - y_j)
\end{equation}
$$

where

$$
\delta_{ij} = \begin{cases}
1 & \text{if } i = j\\
0 & \text{otherwise}
\end{cases}
$$

Now with that out of the way, let's begin.

## Classification

Let's say we are training a classification network with $c$ number of classes. We have a network which outputs a vector $\mathbf{u}$ with $c$ elements. We apply softmax on vector $\mathbf{u}$ to get $\mathbf{y}$ which is the probability of each class.

We get the final output $l = \arg\max(\mathbf{y})$.

When $l$ is fed into the feedback system, we will get either 1 if the prediction was correct or 0 if it was wrong.

**Case 1: Positive Feedback**

If the feedback is correct, we want to minimize the negative log-likelihood, which will be our loss:

$$
\mathcal{L}_p = -\ln(p(l \mid \mathbf{x}))
$$

and in our case $p(l \mid \mathbf{x}) = y_l$.
Now let's compute $\frac{\partial}{\partial u_i} \mathcal{L}_p$.

$$
\begin{align*}
\frac{\partial \mathcal{L}_p}{\partial u_i}  &= \sum_{j=1}^c \frac{\partial \mathcal{L}_p}{\partial y_j} \frac{\partial y_j}{\partial u_i}\\
&= \sum_{j=1}^c \frac{\partial (-\ln(y_l))}{\partial y_j}  y_j(\delta_{ij} - y_i)\\
&= \frac{\partial (-\ln(y_l))}{\partial y_l}  y_l(\delta_{il} - y_i)\\
&= \frac{-1}{y_l} y_l(\delta_{il} - y_i)\\
&= y_i - \delta_{il}
\end{align*}
$$

$$
\begin{equation}
\frac{\partial \mathcal{L}_p}{\partial u_i} =
\begin{cases}
y_i - 1 \leq 0 & \text{if } i = l\\
y_i \geq 0 & \text{otherwise}
\end{cases}
\end{equation}
$$

So it will increase the value of $u_l$ which in turn makes the prediction more confident and decreases for all other cases.

**Case 2: Negative Feedback**

If the feedback is negative,

$$
\mathcal{L}_p = -\ln(p(l^c \mid \mathbf{x})) = -\ln(1 - p(l \mid \mathbf{x}))
$$

and in our case $p(l \mid \mathbf{x}) = y_l$.
Now let's compute $\frac{\partial}{\partial u_i} \mathcal{L}_p$.

$$
\begin{align*}
\frac{\partial \mathcal{L}_p}{\partial u_i} &= \sum_{j=1}^c \frac{\partial \mathcal{L}_p}{\partial y_j} \frac{\partial y_j}{\partial u_i}\\
&= \sum_{j=1}^c \frac{\partial(-\ln(1 - y_l))}{\partial y_j} y_j (\delta_{ij} - y_i)\\
&= \frac{\partial (-\ln(1 - y_l))}{\partial y_l}  y_l (\delta_{il} - y_i) \\
&= \frac{1}{1 - y_l} y_l(\delta_{il} - y_i)
\end{align*}
$$

$$
\begin{equation}
\frac{\partial \mathcal{L}_p}{\partial u_i} =
\begin{cases}
y_l \geq 0 & \text{if } i = l\\
\frac{-y_l}{1 - y_l} y_i \leq 0 & \text{otherwise}
\end{cases}
\end{equation}
$$

So it will try to decrease $u_l$ which in turn makes it less confident and increase for all other cases. Note that the increase is proportional to current confidence. So the next-best will get the highest increase.

## Temporal Classification

In a temporal classification, [CTC loss](ctc/ctc.md) is the go-to optimization loss. For an input $\mathbf{x}$ and a label $\mathbf{l}$, we define the probability of that label as the sum of probability of all possible alignments of the label in time-steps $T$. We produce an output matrix $U$ for every time-step $t$ and class label $k$. At each time-step we softmax along the characters to get the probability matrix $Y$.

Using this probability matrix, we will decode a sequence $\mathbf{l}$.
The probability for a labelling $\mathbf{l}$ is defined as:

$$
\begin{equation}
p(\mathbf{l}\mid \mathbf{x}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l})} p(\pi \mid \mathbf{x})
\end{equation}
$$

It is also defined as:

$$
\begin{equation}
p(\mathbf{l} \mid \mathbf{x}) = \sum_{s=1}^{\lvert \mathbf{l}^\prime \rvert} \frac{\alpha_t(s) \beta_t(s)}{y^t_{\mathbf{l}^\prime_s}}
\end{equation}
$$

where $\alpha_t(s) = \overline{\alpha}_t(s) y^t_{\mathbf{l}^\prime_s}$ and $\beta_t(s) = \overline{\beta}_t(s) y^t_{\mathbf{l}^\prime_s}$.

Let's pre-compute some gradients which will be useful in the following sections:

$$
\begin{align*}
\frac{\partial p}{\partial y^t_k} &= \sum_{s \in lab(\mathbf{l}, k)} \frac{\partial}{\partial y^t_k} \frac{\alpha_t(s) \beta_t(s)}{y^t_k} = \sum_{s \in lab(\mathbf{l}, k)} \frac{\partial}{\partial y^t_k} \overline{\alpha}_t(s) \overline{\beta}_t(s) y^t_k\\
&= \sum_{s \in lab(\mathbf{l}, k)} \overline{\alpha}_t(s) \overline{\beta}_t(s) = \frac{1}{(y^t_k)^2} \sum_{s \in lab(\mathbf{l}, k)} \alpha_t(s) \beta_t(s)
\end{align*}
$$

$$
\begin{align*}
\frac{\partial p}{\partial u^t_k} &= \sum_{i \in A^\prime} \frac{\partial p}{\partial y^t_i} \frac{\partial y^t_i}{\partial u^t_k} = \frac{\partial p}{\partial y^t_k} (y^t_k(1 - y^t_k)) + \sum_{i \in A^\prime - \{k\}} \frac{\partial p}{\partial y^t_i} (-y^t_i y^t_k)\\
&= \frac{\partial p}{\partial y^t_k} y^t_k + \sum_{i \in A^\prime} \frac{\partial p}{\partial y^t_i} (-y^t_i y^t_k) = \frac{\partial p}{\partial y^t_k} y^t_k + \sum_{i \in A^\prime} \frac{1}{(y^t_i)^2} (-y^t_i y^t_k) \sum_{s \in lab(\mathbf{l}, i)} \alpha_t(s) \beta_t(s)\\
&= \frac{\partial p}{\partial y^t_k} y^t_k - y^t_k \sum_{i \in A^\prime} \frac{1}{y^t_i} \sum_{s \in lab(\mathbf{l}, i)} \alpha_t(s) \beta_t(s) = \frac{\partial p}{\partial y^t_k} y^t_k - y^t_k \sum_{s=1}^{\lvert \mathbf{l}^\prime \rvert} \frac{\alpha_t(s) \beta_t(s)}{y^t_{\mathbf{l}^\prime_s}}\\
&= y^t_k \frac{1}{(y^t_k)^2} \sum_{s \in lab(\mathbf{l}, k)} \alpha_t(s) \beta_t(s) - y^t_k p(\mathbf{l} \mid \mathbf{x}) = \frac{1}{y^t_k} \sum_{s \in lab(\mathbf{l}, k)} \alpha_t(s) \beta_t(s) - y^t_k p(\mathbf{l} \mid \mathbf{x})
\end{align*}
$$

**Case 1: Positive Feedback**

If the feedback is positive, our loss function becomes:

$$
\mathcal{L}_p = -\ln(p(\mathbf{l} \mid \mathbf{x}))
$$

Let's compute $\frac{\partial \mathcal{L}_p}{\partial u^t_k}$:

$$
\begin{equation}
\begin{aligned}
\frac{\partial \mathcal{L}_p}{\partial u^t_k} &= \frac{-1}{p(\mathbf{l} \mid \mathbf{x})} \frac{\partial p}{\partial u^t_k}\\
&= \frac{-1}{p(\mathbf{l} \mid \mathbf{x})} \left(\frac{1}{y^t_k} \sum_{s \in lab(\mathbf{l}, k)} \alpha_t(s) \beta_t(s) - y^t_k p(\mathbf{l} \mid \mathbf{x})\right)\\
&= y^t_k - \frac{1}{y^t_k p(\mathbf{l} \mid \mathbf{x})} \sum_{s \in lab(\mathbf{l}, k)} \alpha_t(s) \beta_t(s)
\end{aligned}
\end{equation}
$$

**Case 2: Negative Feedback**

If the feedback is negative, the loss function becomes:

$$
\mathcal{L}_p = -\ln(1 - p(\mathbf{l} \mid \mathbf{x}))
$$

Let's compute $\frac{\partial \mathcal{L}_p}{\partial u^t_k}$:

$$
\begin{equation}
\begin{aligned}
\frac{\partial \mathcal{L}_p}{\partial u^t_k} &= \frac{-1}{1 - p(\mathbf{l} \mid \mathbf{x})} \frac{-\partial p(\mathbf{l} \mid \mathbf{x})}{\partial u^t_k}\\
&= \frac{1}{1 - p(\mathbf{l} \mid \mathbf{x})} \left[\frac{1}{y^t_k} \sum_{s \in lab(\mathbf{l}, k)} \alpha_t(s) \beta_t(s) - y^t_k p(\mathbf{l} \mid \mathbf{x})\right]
\end{aligned}
\end{equation}
$$
