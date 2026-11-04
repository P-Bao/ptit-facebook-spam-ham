import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import FastText


class FastTextVectorizer(BaseEstimator, TransformerMixin):
    """Học FastText (skip-gram + subword) trên tập train, biểu diễn câu bằng mean / TF-IDF-weighted mean."""
    def __init__(self, vector_size=100, window=5, min_count=2, epochs=10, sg=1,
                 min_n=3, max_n=5, bucket=100_000, pooling="tfidf", seed=42, workers=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.sg = sg
        self.min_n = min_n
        self.max_n = max_n
        self.bucket = bucket
        self.pooling = pooling
        self.seed = seed
        self.workers = workers

    def fit(self, X, y=None):
        if self.pooling not in {"mean", "tfidf"}:
            raise ValueError("pooling phải là 'mean' hoặc 'tfidf'")
        docs = [str(s).split() for s in X]
        model = FastText(
            sentences=docs, vector_size=self.vector_size, window=self.window,
            min_count=self.min_count, epochs=self.epochs, sg=self.sg,
            min_n=self.min_n, max_n=self.max_n, bucket=self.bucket,
            seed=self.seed, workers=self.workers
        )
        # Chỉ giữ KeyedVectors (wv), giải phóng bảng trọng số huấn luyện syn1neg để giảm ~50% RAM/Disk
        self.wv_ = model.wv
        del model

        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf_ = {t: np.log((1.0 + n) / (1.0 + c)) + 1.0 for t, c in df.items()}
        self.default_idf_ = np.log(1.0 + n) + 1.0
        return self

    def transform(self, X):
        wv = self.wv_
        out = np.zeros((len(X), self.vector_size), dtype=np.float32)
        use_tfidf = (self.pooling == "tfidf")
        for i, s in enumerate(X):
            toks = str(s).split()
            if not toks:
                continue
            counts = Counter(toks)
            uniq = list(counts.keys())
            V = np.vstack([wv[t] for t in uniq])  # FastText sinh được vector cho cả từ OOV nhờ subword
            if use_tfidf:
                w = np.array([counts[t] * self.idf_.get(t, self.default_idf_) for t in uniq], dtype=np.float32)
            else:
                w = np.array([counts[t] for t in uniq], dtype=np.float32)
            out[i] = (w[:, None] * V).sum(axis=0) / w.sum()
        return out

