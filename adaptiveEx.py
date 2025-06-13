import numpy as np
import hnswlib
import lmdb
import math
from collections import deque

class DynamicMemoryController:
    def __init__(self, dim=256, max_size=10000):
        # Memory stores
        self.env = lmdb.open('./memory', map_size=10**9)
        self.index = hnswlib.Index(space='l2', dim=dim)
        self.index.init_index(max_elements=max_size, ef_construction=200, M=16)
        
        # Adaptive learning curves
        self.memory_curve = MemoryUsageCurve()
        self.batch_curve = BatchLearningCurve()
        
        # Memory state tracking
        self.access_count = np.zeros(max_size, dtype=np.uint32)
        self.relevance = np.ones(max_size, dtype=np.float32)
        self.last_accessed = np.zeros(max_size, dtype=np.float32)
        
        # System parameters
        self.time = 0.0
        self.decay_factor = 0.85
        self.consolidation_threshold = 0.7

    def retrieve(self, query_embed, context):
        """Adaptive retrieval based on learning curve"""
        # Calculate optimal k using learning curve
        k = self.memory_curve.optimal_k(self.time)
        
        # Retrieve with adaptive exploration
        ids, distances = self.index.knn_query(query_embed, k=k)
        return self._apply_context_gating(ids[0], distances[0], context)

    def store(self, embedding, experience, importance=1.0):
        """Store with adaptive consolidation"""
        # Project to hyperbolic space
        hyp_embed = self._hyperbolic_project(embedding)
        
        # Store in memory index
        idx = self.index.add_items(hyp_embed.reshape(1, -1))
        
        # Update memory state
        current_time = self._get_time()
        with self.env.begin(write=True) as txn:
            txn.put(f'exp_{idx}'.encode(), experience.encode())
        
        # Update relevance and access tracking
        self.relevance[idx] = importance
        self.last_accessed[idx] = current_time
        
        # Consolidate if needed
        if self._should_consolidate():
            self.consolidate_memory()

    def process_batch(self, experiences):
        """Adaptive batching based on learning curve"""
        batch_size = self.batch_curve.optimal_batch_size(
            self.time, 
            len(experiences),
            np.mean([e['complexity'] for e in experiences])
        )
        
        # Process in optimal batches
        for i in range(0, len(experiences), batch_size):
            batch = experiences[i:i+batch_size]
            self._process_batch(batch)

    def _process_batch(self, batch):
        """Core batch processing with memory integration"""
        # 1. Encode batch
        embeddings = [self._encode(exp['content']) for exp in batch]
        
        # 2. Retrieve relevant context
        context_vectors = [self.retrieve(emb, exp['context']) for emb in batch]
        
        # 3. Generate predictions
        predictions = [self.predict(emb, ctx) for emb, ctx in zip(embeddings, context_vectors)]
        
        # 4. Update memory
        for exp, pred in zip(batch, predictions):
            importance = self._calculate_importance(exp, pred)
            self.store(embeddings[i], exp, importance)
        
        # 5. Update learning curves
        self.memory_curve.update(len(batch), np.mean([p['accuracy'] for p in predictions]))
        self.batch_curve.update(batch_size, len(batch), processing_time)

    def consolidate_memory(self):
        """Memory optimization with quality metrics"""
        # 1. Relevancy decay: r_i = r_i * e^{-λΔt}
        time_diff = self.time - self.last_accessed
        self.relevance *= np.exp(-0.02 * time_diff)
        
        # 2. Remove low-relevance memories
        to_remove = np.where(self.relevance < 0.01)[0]
        for idx in to_remove:
            self._remove_memory(idx)
        
        # 3. Rebuild index with remaining memories
        self._rebuild_index()
        
        # 4. Update consolidation threshold
        self.consolidation_threshold *= 0.95  # Gradually increase consolidation frequency

    # ... (hyperbolic projection, context gating, etc. from previous implementation)
```
