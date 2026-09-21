# HMM Specification

Here, we provide the specification of prior distributions for the `pyro`-based HMMs. This specification is referenced in the paper within footnote 3. We use partially informed Dirichlet priors for the HMM model parameters of initial state distribution $\pi$, the transition probability matrix $A$ and the emission probabilities $B^{Activity}$ for channel `Activity`.

## Prior specification for HMM parameters

We use Dirichlet priors for the HMM model parameters $\pi$, $A$ and $B^{Activity}$. Within model training, we then sample the model parameters from a Categorical distribution.

The remainder of signal channels follows distributional characteristics for the prior and likelihood according to the following schema (data distribution assumption $\rightarrow$ prior distribution assumptions of parameters:

- $Categorical(p) \rightarrow Dirichlet(p')$
- $N(\mu, \sigma) \rightarrow \mu \sim N(0, 1), \sigma \sim HalfNormal(0, 1)$ 
- $Gamma(\alpha, \theta) \rightarrow \alpha \sim LogNormal(0, 1), \theta \sim LogNormal(0, 1)$
- $ZeroInflatedGamma(z, \alpha, \theta) \rightarrow z \sim Bernoulli(\pi_{zero}), \alpha \sim LogNormal(0, 1), \theta \sim LogNormal(0, 1)$

### Initial distribution

The initial distribution receives a strong prior signal in form of a Dirichlet prior that carries a strong belief about the initial state of a sequence being the first hidden state of $n$ hidden states:

$$
\pi_{1 \times n} =
\begin{bmatrix}
\alpha \\
\epsilon \\
\dots \\
\epsilon
\end{bmatrix}^T
$$

The resulting initial distribution is then sampled from a $Categorical$ distribution based on the prior $Dirichlet$ distribution.

### Transition probabilities

We impose a left-to-right transition matrix by specifying an informative Dirichlet prior on the transition probabilities of our HMM:

$$
A_{n \times n} =
\begin{bmatrix}
\epsilon & \frac{\alpha}{2} & \epsilon & \epsilon & \dots & \frac{\alpha}{2} \\
\epsilon & \frac{\alpha}{3} & \frac{\alpha}{3} & \epsilon & \dots & \frac{\alpha}{3} \\
\vdots & \ddots & \ddots & \ddots & \ddots & \vdots \\
\vdots & \dots & \epsilon & \frac{\alpha}{3} & \frac{\alpha}{3} & \frac{\alpha}{3} \\
\vdots & \dots & \dots & \epsilon & \frac{\alpha}{2} & \frac{\alpha}{2} \\
\epsilon & \dots & \dots & \dots & \epsilon & \alpha
\end{bmatrix}
$$

The resulting transition probabilities are then sampled from a $Categorical$ distribution based on the prior $Dirichlet$ distribution.

### Emission probabilities

We provide an informative Dirichlet prior for the emission probabilities per state for channel `Activity` used inside our HMMs. We impose a strong belief that the padded `START` and `END` activities from the set of all activities $\mathcal{A}$ are bound to the first hidden state, resp. the last hidden state:

$$
B^{Activity}_{n \times |\mathcal{A}|} =
\begin{bmatrix}
\alpha & \epsilon & \dots & \epsilon \\
\epsilon & \epsilon & \dots & \epsilon \\
\vdots & \vdots & \ddots & \vdots \\
\epsilon & \dots & \epsilon & \alpha
\end{bmatrix},
\quad \text{where }
\begin{matrix}
\mathcal{A}_0 = \text{START} \\
\mathcal{A}_{|\mathcal{A}|} = \text{END}
\end{matrix}
$$

The resulting samples for the emissions are then sampled from a $Categorical$ distribution. The remainder of signal channels receives an uninformed prior distribution as specified above subject to their data distributional assumptions.

## Distributional assumptions per signal channel

### Activity
$$B^{Activity} \sim Categorical(p_{Activity})$$

### Resource
$$B^{Resource} \sim Categorical(p_{Resource})$$

### Time since last event (TSLE)

$$Z_{TSLE} \sim Bernoulli(\pi_{TSLE})$$

$$B^{TSLE} \sim \begin{cases}0 & \text{,if} \quad Z_{TSLE} = 0\\
    Gamma(\alpha_{TSLE}, \beta_{TSLE}) & \text{,if} \quad Z_{TSLE}=1\end{cases}$$

### Time since case start (TSCS)

$$Z_{TSCS} \sim Bernoulli(\pi_{TSCS})$$

$$B^{TSCS} \sim \begin{cases}0 & \text{,if} \quad Z_{TSCS}=0\\Gamma(\alpha_{TSCS}, \beta_{TSCS}) & \text{,if} \quad Z_{TSCS}=1\end{cases}$$