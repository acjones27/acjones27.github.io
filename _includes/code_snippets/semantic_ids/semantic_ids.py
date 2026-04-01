import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Step 1: encode items into embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

items = [
    "Attention is all you need",
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-3: Language Models are Few-Shot Learners",
    "Neural Collaborative Filtering",
    "Wide & Deep Learning for Recommender Systems",
]

embeddings = model.encode(items, normalize_embeddings=True).astype(np.float32)


# Step 2: residual quantisation — two levels, codebook size 8 each
def train_codebook(vecs, n_centroids=8):
    d = vecs.shape[1]
    kmeans = faiss.Kmeans(d, n_centroids, niter=20, verbose=False)
    kmeans.train(vecs)
    _, indices = kmeans.index.search(vecs, 1)
    return indices.flatten(), kmeans.centroids


def rq_encode(embeddings, n_levels=2, n_centroids=8):
    residual = embeddings.copy()
    all_codes = []
    for _ in range(n_levels):
        codes, centroids = train_codebook(residual, n_centroids)
        all_codes.append(codes)
        residual = residual - centroids[codes]  # subtract and encode what's left
    return np.stack(all_codes, axis=1)


codes = rq_encode(embeddings, n_levels=2, n_centroids=8)

print("Semantic IDs:")
for item, code in zip(items, codes):
    print(f"  {code.tolist()}  ←  {item}")
