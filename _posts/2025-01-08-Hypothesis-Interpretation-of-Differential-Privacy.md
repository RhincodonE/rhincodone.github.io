---
layout: post
title: Hypothesis Interpretation of Differential Privacy
date: 2025-1-8
description: In this post, I'll generally introduce the hypothesis testing interpretation of differential privacy and why it's vital for private machine learning.
tags: DP-hp
categories: Differential privacy
related_posts: false
---

# Hypothesis Interpretation of Differential Privacy

##Content

- **Basic definition of differential privacy**
- **Hypothesis interpretation of differential privacy**
- **Membership inference attack (MIA) and hypothesis testing**
- **Auditing differential privacy with MIA**
- **Conclusion**

---

##Basic definition of differential privacy
Let's start with the basic definition of differential privacy. Specifically, we only focus on the widely adopted relaxed $$(\epsilon,\delta)-\text{differential privacy}$$.

**Definition 1:** $$M$$ is $$(\epsilon,\delta)-\text{differential privacy}$$ if, for all neighboring data $$D_0$$ and $$D_1$$ that differ in at most one sample, and for all measurable sets $$\mathcal{O}$$,

$$
P(M(D_0)\in \mathcal{O}) \leq e^\epsilon P(M(D_1)\in \mathcal{O}) + \delta
$$

and symmetrically,

$$
P(M(D_1)\in \mathcal{O}) \leq e^\epsilon P(M(D_0)\in \mathcal{O}) + \delta
$$

These symmetric conditions highlight that differential privacy mathematically defines the "indistinguishability" of the output of $$M(D_0)$$ and $$M(D_1)$$ on the measurable sets $$\mathcal{O}$$ (output space).

This "indistinguishability", a binary classification problem, can be naturally described by hypothesis testing.

---

## Hypothesis interpretation of differential privacy

Based on the "indistinguishablity" on the $$M(D_0)$$ and $$M(D_1)$$, we set up two hypothesis,

$$
H_0: \text{M used } D_0
\tag{DP: Null hypothesis}
$$

$$
H_1: \text{M used } D_1
\tag{DP: Alternative hypothesis}
$$

Now let's interpret what benefits we can get from it. We start with two errors in the basic hypothesis testing.

### Type I Error (significants level
This occurs when the null hypothesis ($$H_0$$) is **rejected** even though it is true, i.e., M is predicted did not use $$D_0$$ when it actually did. The probability of this error is defined as:

$$
\alpha = P_{\text{FA}}(D_0, D_1, M, S) \equiv P(M(D_0) \in S)
\tag{1}
$$

where $$S$$ is the rejection region for $$H_0$$.

Type I error is also called "significance level." It represents the maximum probability of making a Type I error that we are willing to tolerate in a test. When we say a test is conducted at the $$\alpha$$ significance level, we mean:

$$
P(\text{Reject } H_0|H_0 \text{ is true}) = \alpha
$$

That is, if we repeated the same test many times under conditions where $$H_0$$ is actually true, we'd expect to falsely reject $$H_0$$ about $$\alpha \times 100\%$$ of time. Thus, $$\alpha$$ control how "strict" our test is in claiming a significant result

### Type II Error
This occurs when the null hypothesis ($$H_0$$) is **not rejected** even though it is false, i.e., M is predicted to use $$D_0$$ when it actually did not. The probability of this error is defined as:

$$
\beta = P_{\text{MD}}(D_0, D_1, M, S) \equiv P(M(D_1) \in \bar{S}),\
\tag{2}
$$

where $$\bar{S}$$ is the complement of the rejection region $$S$$.

Since $$\beta$$ is the probability of mistakenly accepting the null hypothesis, $$1-\beta$$ represents the correctly accepting the null hypothesis. In other words, if there really is a difference or effect to be found, a test with high power ($$1-\beta$$) is more likely to detect it. Thus, we also term $$1-\beta$$ as the "power" of hypothesis testing.


### Differential Privacy in Terms of Hypothesis Testing

Here we turn to why we want to use a hypothesis to represent differential privacy. Actually, the hypothesis interpretation of differential privacy condition imposes constraints on the probabilities of **false alarms** (Type I errors) and **missed detections** (Type II errors). Let's see how:

For a mechanism $$M$$ to satisfy $$(\epsilon, \delta)$$-differential privacy, the following conditions must hold for all neighboring databases $$D_0$$ and $$D_1$$, and for all rejection regions $$S$$:

1. $$
   \alpha + e^\epsilon \beta \geq 1 - \delta
   \tag{3}
   $$

2. $$
   e^\epsilon \alpha + \beta \geq 1 - \delta
   \tag{4}
   $$

3. $$
   \alpha \leq e^\epsilon(1-\beta) + \delta
   \tag{5}
   $$
4. $$
   1 -\beta \leq e^\epsilon \alpha + \delta
 	\tag{6}
	$$

**Proof:**

Assuming $$M$$ is $$(\epsilon,\delta)-\text{differentially private}$$ on neigboring datasets $$D_0$$ and $$D_1$$, and the complement sets $$\bar{S}$$ of rejection region $$$$, we have:

$$
P(M(D_0)\in \bar{S}) \leq e^\epsilon P(M(D_1)\in \bar{S}) + \delta
\tag{7}
$$

and

$$
P(M(D_1)\in \bar{S}) \leq e^\epsilon P(M(D_0)\in \bar{S}) + \delta
\tag{8}
$$

Combining equations 1, 2, 7, and 8, we can derive equations 3, and 4.

Since $$\bar{S}$$ is the complement set of rejection region $$S$$, and $$\bar{S}$$ is a measurable sets. $$S$$ is also a measurable set. Thus equations 7, and 8 are also valid on $$S$$:

$$
P(M(D_0)\in S) \leq e^\epsilon P(M(D_1)\in S) + \delta
\tag{9}
$$

and

$$
P(M(D_1)\in S) \leq e^\epsilon P(M(D_0)\in S) + \delta
\tag{10}
$$

Combining equations 1, 2, 9, and 10, we can derive equations 5, and 6.




###Privacy Region

Now we can define a privacy region for $$(\epsilon,\delta)-\text{differential privacy}$$ as :

$$
\mathcal{R}(\epsilon , \delta) \equiv \{(\alpha,\beta)|\alpha + e^\epsilon \beta \geq 1 - \delta, \text{ and }e^\epsilon \alpha + \beta \geq 1 - \delta,\text{ and }\alpha \leq e^\epsilon(1-\beta) + \delta,\text{ and } 1 -\beta \leq e^\epsilon \alpha + \delta\}
$$

We can also define a privacy region for $$(\epsilon,\delta)-\text{differentially private}$$ mechanism $$M$$ on neigboring datasets $$D_0$$ and $$D_1$$, and any rejection region $$S\subseteq \mathcal{O}$$ as:


$$
\mathcal{R}(M,D_0,D_1) \equiv \text{conv}(\{(\alpha,\beta)|\text{ for all }S\subseteq \mathcal{O}\})
$$

Conv is the convex hull of a point set.

---

## Membership inference attack (MIA) and hypothesis testing

### What are membership inference attacks (MIA)

Membership inference attacks aim to distinguish if a given data sample(s) is in the training set of a target model. It's a typical binary distinguishing attack. MIA usually classifies a sample as "member" or "non-member" which represents whether the sample is training data or not. Typically, "Member" is a positive prediction, and "non-member" is a negative prediction.

### Differential privacy and MIA

Since MIAs are distinguishing attacks, we can also use the hypothesis to perform the attacking target:

Assuming mechanism $$M$$ is a machine-learning model, victim sample is $$x$$,

$$
H_0: M\text{ is not trained on } x
\tag{MIA: Null hypothesis}
$$

$$
H_1: M \text{ is trained on } x
\tag{MIA: Alternative hypothesis}
$$

Assuming $$x\in D_1$$ and $$x\notin D_0$$, that is, $$D_0$$ and $$D_1$$ are neighboring datasets that differ only in $$x$$. Then we compare the current hypothesizes in MIA with the hypothesizes in differential privacy (equation DP: Null hypothesis and DP: Alternative hypothesis), and we found that both the null hypothesizes represent the same thing - $$M$$ is not trained on $$x$$, and so do both alternative hypothesizes.

### Type I and Type II errors in MIA

Now we revisit the definitions of Type I and II errors and corresponding equations. We know that for the hypothesis-based membership inference attack:

$$
\text{FPR} = \alpha = P_{\text{FA}}(D_0, D_1, M, S) \equiv P(M(D_0) \in S)
\tag{11}
$$

$$
\text{FNR} = \beta = P_{\text{MD}}(D_0, D_1, M, S) \equiv P(M(D_1) \in \bar{S}),\
\tag{12}
$$

For simplicity we show the relationship between the confusion matrix and type I and II errors in the following table:

| Metric                     | Definition                                                                                         | Error Type          |
|----------------------------|----------------------------------------------------------------------------------------------------|---------------------|
| **True Positive Rate (TPR)** | $$\text{TPR} = 1 - \text{FNR}$$                                                                   | Correct detection  $$1-\beta$$ |
| **False Positive Rate (FPR)** | $$\text{FPR} = P(M(D_0) \in S)$$                                                                | Type I Error $$\alpha$$       |
| **False Negative Rate (FNR)** | $$\text{FNR} = P(M(D_1) \in \bar{S})$$                                                          | Type II Error $$\beta$$      |
| **True Negative Rate (TNR)** | $$\text{TNR} = 1 - \text{FPR}$$                                                                   | Correct rejection $$1-\alpha$$  |

### MIA audits implementation of differential privacy

According to this table:

1. $$
   \alpha + e^\epsilon \beta \geq 1 - \delta \iff \text{FPR}+e^\epsilon \text{FNR} \geq 1 - \delta \iff \frac{1-\text{FPR}-\delta}{1-\text{TPR}}\leq e^\epsilon
   \tag{13}
   $$

2. $$
   e^\epsilon \alpha + \beta \geq 1 - \delta \iff e^\epsilon \text{FPR} + \text{FNR} \geq 1 - \delta \iff \frac{\text{TPR}-\delta}{\text{FPR}} \leq e^\epsilon
   \tag{14}
   $$

3. $$
   \alpha \leq e^\epsilon(1-\beta) + \delta \iff \text{FPR} \leq e^\epsilon(1-\text{FNR}) + \delta \iff \frac{\text{FPR}-\delta}{\text{TPR}} \leq e^\epsilon
   \tag{15}
   $$

4. $$
   1 -\beta \leq e^\epsilon \alpha + \delta \iff 1- \text{FNR} \leq e^\epsilon \text{FPR} +\delta \iff \frac{\text{TPR}-\delta}{\text{FPR}} \leq e^\epsilon
 	\tag{16}
	$$

We can find that equation 14 and 16 are equivalent. When we draw the privacy region, we can use equations 13, 15, and one from 14 and 16.

With these three inequality, or namely the lower bound of $\epsilon$, we can audit the implementation of differential privacy. But why do we need to do auditing? In the recent couple of years, with the flourishing of machine learning, differential privacy has been incorporated into machine learning models to make a trained model differentially private. However, machine learning models are typically black-box, we don't really know what happens inside when we train a model, let go of how epsilon changes through the entire training. To debug the potential implementation mistakes when training a differentially private model, we need to use some tool to audit the implementation.

In the next section, I'll introduce the basic concept of auditing differential privacy.

---

##Auditing differential privacy with MIA

Basically, auditing differential privacy aims to use membership inference attacks to attack the target model and get the TPR, and FPR on each sample. Then we compute the lower bounds in equations 13, 15, and one of 14 and 16. If one of the lower bounds is not valid, then the implementation of differential privacy may have some bugs.

As a simple example, you can try this [app](https://huggingface.co/spaces/Rhincodon/privacy-region-example).This app plots the privacy region of differential privacy on Type I & Type II error rates (FPR and 1-TPR). You can try to understand the differential privacy by adjusting the $$\epsilon$$ and $$\delta$$ and inspecting the area of privacy region. Typically, smaller $$\epsilon$$ will make a smaller privacy region and allow less combination of $$(TPR,FPR)$$. You can also try different points of $$(TPR,FPR)$$ and check if they lie in the privacy area. If a green dot shows, it indicates the $$TPR$$ and $$FPR$$ is valid for all equations 13 - 16. If it's a red point, the $$TPR$$ and $$FPR$$ make at least one of equation 13-16 invalid.

### Problem in simple auditing

A problem in the simplest method upon is that due to the empirical estimation of TPR, FPR are on a finite number of models, the uncertainty is inevitable if we don't have statistical results or estimation of TPR and FPR.

Imagine you're conducting a **privacy audit** on a differentially private algorithm $$ M $$. The goal is to empirically estimate the privacy parameters $$ \varepsilon $$ and $$ \delta $$ by assessing an adversary's ability to distinguish whether $$ M $$ was trained on dataset $$ D_0 $$ or a neighboring dataset $$D_1$$. Specifically, you aim to estimate:

**Type I Error ($$ \alpha $$)**

**Type II Error ($$ \beta $$)**

To **accurately bound** $$ \alpha $$ and $$ \beta $$, ensuring the privacy guarantees ($$ \varepsilon $$, $$ \delta $$) hold with high confidence. This requires not just single estimates of TPR and FPR but **statistical bounds** that account for uncertainty due to finite sample sizes.

Let's take a simple example:

You got these two estimated TPR and FPR after running MIA on multiple DP models where $$\epsilon = 2.5$$, $$\delta = 0.0001$$:

**$$\text{Estimated  } FPR (\alpha) = 10\% $$**
**$$\text{Estimated  } TPR (1-\beta) = 90\% $$**

Based on these single estimates, you might conclude that:

$$\frac{TPR-\delta}{FPR} \approx 9 < e^\epsilon \approx 12.18$$

well, the implementation of differential privacy has no bugs.

But we all know that the model training has uncertainty and finite hypothesis testing also has uncertainty. How confident can we say that the next estimation will not show the implementation has bugs?

What we need is confidence. For example, with 95\% confidence, the estimated results will stay in the privacy region.

In the next blog, I'll introduce how to bound TPR and FPR with Clopper-pearson interval. This mathematical tool will give us a range of TPR and FPR with a specific confidence level. Through this bound, we can not only debug the implementation of differential privacy, we can also see how well the differential privacy is implemented.
