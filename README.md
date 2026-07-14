# BanditOptimize: Dynamic Experimentation Engine

An end-to-end framework comparing traditional Static A/B Testing against Bayesian Multi-Armed Bandits (Thompson Sampling) for real-time traffic routing and conversion optimization.

## Repo Intro
Traditional A/B testing relies on a static "explore-then-exploit" framework. This forces a company to waste high volumes of traffic on underperforming variants during the exploration phase, resulting in high "Regret" (lost revenue). 

This repository implements a **Thompson Sampling Bandit**, which uses Bayesian updating to dynamically route traffic to the winning variant in real-time, minimizing regret and capturing conversions that would be lost during a traditional A/B test.

##  Mathematical Foundations
The model utilizes a Beta-Binomial conjugate prior. For each arm $i$, the probability of conversion $\theta_i$ is modeled as a Beta distribution:
$$\theta_i \sim Beta(\alpha_i, \beta_i)$$

Upon observing a reward (conversion $= 1$, no conversion $= 0$), the posterior is updated instantly:
$$Beta(\alpha_i + \text{reward}, \beta_i + (1 - \text{reward}))$$

##  Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run Pipeline
   ```bash
   python -m bandit_optimize
   ```