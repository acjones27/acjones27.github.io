---
title: "Choice Models: How Do People Actually Decide?"
date: 2026-03-17
categories: [RecSys]
tags: [recsys, ranking]
---

When someone clicks on a search result, they made a choice. Modelling that choice well turns out to be surprisingly non-trivial.

Choice models are a family of statistical models, originally from economics, that try to describe how people select between options. They show up in recommender systems constantly — often without anyone realising it.

---

## The basic idea

The most common choice model is called the **Multinomial Logit model** (or MNL for short). The idea is that each item has some *utility* to a user — a score representing how appealing it is — and the probability of choosing a particular item depends on that score relative to everything else available.

Concretely: imagine a user is shown three items — a dog toy, a cat bed, and a fish tank. Each item gets a utility score based on how relevant it is to that user. Say the scores are:

- 🐶 Dog toy: 2.0
- 🐱 Cat bed: 1.0  
- 🐟 Fish tank: 0.5

The MNL model says: exponentiate each score, then divide by the sum of all of them. So:

- P(dog toy) = e^2.0 / (e^2.0 + e^1.0 + e^0.5) = 7.39 / (7.39 + 2.72 + 1.65) = **63%**
- P(cat bed) = 2.72 / 11.76 = **23%**
- P(fish tank) = 1.65 / 11.76 = **14%**

If you've done any deep learning, you'll recognise this immediately — it's a **softmax**. Every time you apply softmax to a set of scores in a neural network, there's a multinomial logit choice model underneath. The utility score is just whatever your model outputs before the softmax layer.

---

## Where things get interesting: the red bus / blue bus problem

The MNL model has a well-known limitation. It assumes that adding a new option to the set affects all other options equally — which turns out to be unrealistic when options are similar to each other.

The classic illustration: imagine a commuter choosing between a **red bus** and a **car**. They're equally appealing, so it's 50/50. Now a **blue bus** is introduced — identical to the red bus in every way except colour.

Intuitively, the blue bus should steal share from the red bus, not from the car. The split should become something like:
- Car: 50%
- Red bus: 25%
- Blue bus: 25%

But the MNL model gives you 33/33/33 — it treats the blue bus as completely independent from the red bus. It has no concept of "these two options are basically the same thing".

In recommenders, this matters a lot. If you show ten very similar articles on a page, MNL thinks the user is ten times more likely to engage with one of them. In reality, similarity between items on a slate affects choice in ways the basic model can't capture.

One response to this is to stop assuming a specific model altogether and just learn the choice model from data instead. A recent paper from RecSys 2025, LCM4Rec (Krause & Oosterhuis), does exactly that — rather than picking MNL or Nested Logit and hoping the assumption fits, it infers the most likely choice model from the observed behaviour. It's a more honest approach, especially when you're not sure how your users actually make decisions.

---

## How this plays out in practice

**In recommender systems**, this connects to a problem called *slate optimisation*. Instead of ranking items independently and serving the top-k, you can think about the whole set of items together — choosing a slate where the items complement each other rather than compete. This naturally encourages diversity, and models that account for item similarity (like the Nested Logit model, which groups similar items so they compete more with each other) handle this better than vanilla MNL.

**In academic publishing**, choice models are useful for interpreting which papers researchers actually engage with versus which ones they just download. If a paper appeared at position 3 in search results, that's relevant context for interpreting the click — a click at position 3 means something different than a click at position 1. Position is effectively a feature in the utility function, and correcting for it gives much cleaner signal about true relevance.

The same idea shows up in how language models are trained on human preference data — if you're curious about that, [this post by Michael Brenndoerfer](https://mbrenndoerfer.com/writing/bradley-terry-model-pairwise-preferences-rankings) is a clear explanation of how pairwise choices get turned into training signal.

---

## Code: the red bus / blue bus problem

Here's the problem in code. We start with a 50/50 split between car and red bus, add an identical blue bus, and compare what MNL (softmax) predicts vs. a Nested Logit model that knows the buses are similar:

```python
{% include_relative ../code_snippets/choice_model/choice_model.py %}
```

---

## Further reading

- **Train, K. (2009)** — [Discrete Choice Methods with Simulation](https://eml.berkeley.edu/books/choice2.html): the standard textbook, free online. Chapter 2 covers MNL and the red bus / blue bus problem clearly
- **Cao et al. (2022)** — [Slate-Aware Ranking for Recommendation](https://arxiv.org/abs/2210.11403): how to think about optimising a whole slate of items rather than ranking them independently
- **Joachims et al. (2017)** — [Unbiased Learning-to-Rank with Biased Feedback](https://arxiv.org/abs/1608.04468): a good entry point for understanding position bias in search and how to correct for it
- **Krause & Oosterhuis (2025)** — [A Non-Parametric Choice Model That Learns How Users Choose Between Recommended Options](https://dl.acm.org/doi/10.1145/3705328.3748090): the LCM4Rec paper from RecSys 2025, which sidesteps the "which model do I assume?" question entirely by learning the choice model from data
