---
title: "Semantic IDs: Giving Items a Meaningful Address"
date: 2026-03-10
categories: [RecSys]
tags: [recsys, llm, embeddings]
---

Most systems that deal with items — products, papers, songs — represent them with an ID. Something like `item_00482931`. It's unique and stable, but completely opaque. A model that sees this ID has no idea what the item is about.

Semantic IDs try to fix that by assigning items IDs that actually carry meaning, so that similar items get similar IDs.

---

## How they work

The starting point is a content embedding — a vector that encodes what an item is about, generated from its text, metadata, or both. You then compress that embedding into a short sequence of discrete codes using a technique called **RQ-VAE** (Residual Quantization Variational Autoencoder). Don't worry too much about the name — the idea is that you're approximating the embedding in steps, each step capturing what the previous one missed, and representing each step as a code from a fixed list of possible values (called a codebook).

The result is something like `[42, 7, 13]` — a short sequence of integers that represents the item. Items with similar content end up with similar sequences, often sharing a prefix. A paper on BERT might get `[42, 7, 13]` and a paper on GPT might get `[42, 7, 19]` — same first two codes, different third. Items that are far apart in meaning get completely different sequences.

One practical thing to know: collisions are more common than you might expect. With a finite codebook, different items can end up assigned the same ID — and a few levels may not be enough to guarantee uniqueness across a large catalogue.

---

## What do Semantic IDs give you over just using the embedding?

This is the part I found genuinely confusing at first, and I'm still working through it — so take this with a pinch of salt.

The embedding and the Semantic ID both encode the same underlying content information. The cold start argument you sometimes see is partially right but not the full story — you can generate an embedding for a new item just as easily. Where it does matter slightly in the generative setting is that a new item with a similar prefix to existing items can benefit from what the model already learned about those items at inference time, without retraining.

The more compelling argument is about what kind of model you can build with them. Embeddings are continuous vectors — you use them by computing distances or dot products, which is great for retrieval. But you can't ask a language model to *generate* an embedding the way it generates text, token by token.

Semantic IDs are discrete tokens, which means they live in the same space as words. In principle this lets you build a model that's fluent in both natural language and item IDs at the same time — so a user could say "recommend me something similar but for a different audience" and the model could interpret that constraint and generate an appropriate item ID, without a separate intent classifier or routing layer sitting on top. The recommendations and the conversation happen in the same model.

That said, based on what I've read, this comes at a cost to raw recommendation accuracy — you're trading some precision for a richer interface. Whether that's worthwhile depends on what you're building. Eugene Yan has a detailed writeup where he actually builds and evaluates this end to end, including an honest look at that tradeoff — linked in further reading.

From what I can tell, if you're not building a generative recommendation system — if you just want to retrieve similar items — Semantic IDs don't give you much over keeping the embedding directly. I could be wrong about this though, I'm still figuring it out.

---

## Code: building Semantic IDs with RQ-VAE

Here's a minimal example using `sentence-transformers` for the embeddings and `faiss` for the quantisation:

```python
{% include_relative ../code_snippets/semantic_ids/semantic_ids.py %}
```

You should see the transformer papers share a first-level code, and the recommender papers share a different one — that's the hierarchical structure working as intended. With only 5 items and a codebook of size 8 this is a toy example; in practice you'd want far more items than codebook entries, and you'd need to watch for collisions.

---

## If you actually want to use Semantic IDs

The code above is illustrative. For a real implementation, the steps are:

1. **Generate embeddings** for all your items using a pretrained encoder (sentence-transformers, or a domain-specific model if you have one)
2. **Train an RQ-VAE** on those embeddings to learn the codebooks — the example above uses k-means as a simplified stand-in
3. **Check for collisions** and add a sequential tie-breaker level if needed
4. **Extend your language model's vocabulary** with the new ID tokens and fine-tune it on sequences of user behaviour — this is the substantial engineering effort, and where Eugene Yan's post goes into real detail

If you're not going the generative recommendation route, it's worth asking whether a well-built embedding-based retrieval system gets you most of the way there with considerably less complexity.

---

## Further reading

- **Eugene Yan (2025)** — [Training an LLM-RecSys Hybrid for Steerable Recs with Semantic IDs](https://eugeneyan.com/writing/semantic-ids/): the most complete hands-on writeup I've found — he actually builds and evaluates the full system end to end, including the honest performance comparison against a standard baseline
- **TIGER** — [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) (Rajput et al., 2023): the paper that introduced Semantic IDs to the recsys community in this form
- **RQ-VAE** — [Autoregressive Image Generation using Residual Quantization](https://arxiv.org/abs/2203.01941): where the quantisation technique comes from (image generation, but the method transfers directly)
- **LETTER** (2024) — a follow-up that improves Semantic ID construction using contrastive learning, worth reading after TIGER
