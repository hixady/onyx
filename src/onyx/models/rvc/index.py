import numpy as np


class IndexManager:
    def __init__(self, index):
        self.index = None
        self.big_npy = None
        if index is None:
            return
        try:
            import faiss
            import numpy as np
            if isinstance(index, bytes):
                self.index = faiss.deserialize_index(np.frombuffer(index, dtype=np.uint8))
            else:
                self.index = faiss.read_index(index)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
            print(f"  loaded index: {self.index.ntotal} entries, dim={self.index.d}")
        except ImportError:
            print("  faiss not installed, skipping index")
        except Exception as e:
            print(f"  failed to load index: {e}")

    def retrieve(self, features, index_rate=0.5):
        if self.index is None or index_rate <= 0:
            return features
        npy = features.astype(np.float32)
        if npy.shape[1] != self.index.d:
            print(f"  feature dim mismatch: index={self.index.d}, input={npy.shape[1]}, truncating")
            npy = npy[:, :self.index.d]
        score, ix = self.index.search(npy, k=8)
        if (ix < 0).any():
            return features
        weight = np.square(1.0 / np.clip(score, 1e-10, None))
        weight /= weight.sum(axis=1, keepdims=True)
        retrieved = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
        return retrieved * index_rate + features * (1.0 - index_rate)
