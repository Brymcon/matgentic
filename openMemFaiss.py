```python
import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer
import transformers
import torch
import time
import hashlib
import faiss
import re
from collections import Counter
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedCPUMemory:
    def __init__(self, db_path='memory.db', persist=True):
        # Initialize embedding model
        self.embedder = SentenceTransformer(
            "BAAI/bge-m3",
            device='cpu',
            use_auth_token=True
        )
        
        # Initialize quantized LLM
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.7B")
        self.llm = transformers.pipeline(
            "text-generation",
            model="Qwen/Qwen1.5-1.7B",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            model_kwargs={"load_in_4bit": True}
        )
        
        # Database setup
        self.db_path = db_path
        self.persist = persist
        self.memory_db = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        
        # FAISS configuration
        self.dimension = 1024
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimension))
        
        # Adaptive learning parameters
        self.decay_factor = 0.95
        self.cache_size = 4096
        self.emergence_threshold = 0.78
        self.learning_rate = 0.15
        self.relevance_boost = 1.25
        self.id_counter = self._get_max_id() + 1
        
        # Load existing memories
        self._load_existing_memories()

    def _init_db(self):
        """Initialize database schema with persistence support"""
        cur = self.memory_db.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY,
            content TEXT,
            tags TEXT,
            weight REAL DEFAULT 1.0,
            last_accessed REAL,
            links TEXT,
            access_count INTEGER DEFAULT 1,
            semantic_density REAL DEFAULT 0.0,
            embedding BLOB
        )''')
        self.memory_db.commit()

    def _get_max_id(self):
        """Get maximum ID from database for continuation"""
        cur = self.memory_db.cursor()
        cur.execute("SELECT MAX(id) FROM memories")
        max_id = cur.fetchone()[0]
        return max_id if max_id is not None else 0

    def _load_existing_memories(self):
        """Load existing memories from database into FAISS index"""
        cur = self.memory_db.cursor()
        cur.execute("SELECT id, embedding FROM memories")
        results = cur.fetchall()
        
        if not results:
            return
            
        ids = []
        embeddings = []
        
        for mem_id, embedding_blob in results:
            if embedding_blob:
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                embeddings.append(embedding)
                ids.append(mem_id)
        
        if embeddings:
            embeddings = np.vstack(embeddings)
            self.index.add_with_ids(embeddings, np.array(ids))
            logging.info(f"Loaded {len(ids)} memories into FAISS index")

    def add_memory(self, content, context=None):
        """Create memory with contextual awareness and semantic relationships"""
        try:
            # Generate adaptive embedding
            embedding = self._generate_contextual_embedding(content, context)
            
            # Advanced tagging
            tags = self._generate_semantic_tags(content)
            
            # Create memory ID
            mem_id = self.id_counter
            self.id_counter += 1
            
            # Find semantic relationships
            related = self.find_related(embedding, k=3)
            links = [r[0] for r in related]
            
            # Calculate semantic density
            density = self._calculate_semantic_density(embedding, links)
            
            # Store in DB
            cur = self.memory_db.cursor()
            cur.execute('''INSERT INTO memories 
                        (id, content, tags, last_accessed, links, semantic_density, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                        (mem_id, content, ','.join(tags), time.time(), 
                         ','.join(map(str, links)), density, embedding.tobytes()))
            self.memory_db.commit()
            
            # Update index
            self._update_index(mem_id, embedding)
            
            # Adaptive learning update
            self._update_related_weights(links, 0.07)
            
            return mem_id
            
        except Exception as e:
            logging.error(f"Error adding memory: {e}")
            return None

    def _generate_contextual_embedding(self, content, context=None):
        """Context-aware embedding generation with error handling"""
        try:
            if context:
                augmented_content = f"Context: {context}\nMemory: {content}"
                return self.embedder.encode(
                    augmented_content, 
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    precision='float32'
                )
            return self.embedder.encode(
                content, 
                convert_to_tensor=False,
                normalize_embeddings=True,
                precision='float32'
            )
        except Exception as e:
            logging.error(f"Embedding generation failed: {e}")
            return np.zeros(self.dimension, dtype=np.float32)

    def _generate_semantic_tags(self, content):
        """Enhanced semantic tagging with noun extraction"""
        try:
            words = re.findall(r'\b\w{4,}\b', content.lower())
            word_freq = Counter(words)
            return [word for word, _ in word_freq.most_common(5)]
        except:
            return []

    def _calculate_semantic_density(self, embedding, neighbor_ids):
        """Calculate how clustered this memory is among its neighbors"""
        if not neighbor_ids:
            return 0.0
            
        try:
            # Retrieve neighbor embeddings
            neighbor_embeddings = []
            cur = self.memory_db.cursor()
            
            for mem_id in neighbor_ids:
                cur.execute("SELECT embedding FROM memories WHERE id = ?", (mem_id,))
                result = cur.fetchone()
                if result and result[0]:
                    neighbor_emb = np.frombuffer(result[0], dtype=np.float32)
                    neighbor_embeddings.append(neighbor_emb)
            
            if not neighbor_embeddings:
                return 0.0
                
            # Calculate average similarity
            neighbor_embeddings = np.vstack(neighbor_embeddings)
            similarities = np.dot(neighbor_embeddings, embedding)
            return np.mean(similarities)
            
        except Exception as e:
            logging.error(f"Semantic density calculation failed: {e}")
            return 0.0

    def _update_index(self, mem_id, embedding):
        """Update FAISS index with new memory"""
        try:
            embedding = embedding.astype(np.float32).reshape(1, -1)
            self.index.add_with_ids(embedding, np.array([mem_id]))
        except Exception as e:
            logging.error(f"Index update failed: {e}")

    def find_related(self, embedding, k=5, min_similarity=0.65):
        """Find related memories with adaptive thresholding"""
        if self.index.ntotal == 0:
            return []
            
        try:
            # Normalize and search
            embedding = embedding.astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            distances, ids = self.index.search(embedding.reshape(1, -1), k)
            
            # Convert to similarities (since we're using IP)
            similarities = (2.0 - distances) / 2.0
            
            results = []
            for i in range(len(ids[0])):
                if similarities[0][i] > min_similarity:
                    results.append((int(ids[0][i]), float(similarities[0][i])))
                    
            return results
            
        except Exception as e:
            logging.error(f"Similarity search failed: {e}")
            return []

    def retrieve(self, query, k=5):
        """Contextual retrieval with adaptive reinforcement"""
        # Memory management
        if self.index.ntotal > self.cache_size:
            self._adaptive_offload()
            
        try:
            # Generate query embedding
            query_embed = self.embedder.encode(
                query, 
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            
            # Find related memories
            related = self.find_related(query_embed, k=k, min_similarity=0.6)
            
            # Update weights based on relevance
            for mem_id, similarity in related:
                boost = 1.0 + (self.relevance_boost * similarity)
                self._update_weight(mem_id, boost * self.learning_rate)
                
            return [self.get_memory(mid) for mid, _ in related]
            
        except Exception as e:
            logging.error(f"Retrieval failed: {e}")
            return []

    def _update_weight(self, mem_id, delta):
        """Update memory weight with access tracking"""
        try:
            cur = self.memory_db.cursor()
            cur.execute('''UPDATE memories 
                        SET weight = weight * ? + ?,
                        last_accessed = ?,
                        access_count = access_count + 1
                        WHERE id = ?''',
                        (self.decay_factor, delta, time.time(), mem_id))
            self.memory_db.commit()
        except Exception as e:
            logging.error(f"Weight update failed: {e}")

    def _update_related_weights(self, mem_ids, delta):
        """Update weights of related memories"""
        for mem_id in mem_ids:
            self._update_weight(mem_id, delta)

    def _adaptive_offload(self):
        """Intelligent memory offloading based on multiple factors"""
        try:
            cur = self.memory_db.cursor()
            cur.execute('''SELECT id FROM memories 
                        WHERE weight * semantic_density < 0.15
                        ORDER BY last_accessed ASC 
                        LIMIT 50''')
            to_offload = cur.fetchall()
            
            if not to_offload:
                return
                
            offloaded_ids = [row[0] for row in to_offload]
            
            # Remove from index
            self.index.remove_ids(np.array(offloaded_ids, dtype=np.int64))
            
            # Remove from database
            placeholders = ','.join('?' * len(offloaded_ids))
            cur.execute(f'''DELETE FROM memories WHERE id IN ({placeholders})''', offloaded_ids)
            self.memory_db.commit()
            
            logging.info(f"Offloaded {len(offloaded_ids)} low-priority memories")
            
        except Exception as e:
            logging.error(f"Offloading failed: {e}")

    def process_query(self, query):
        """Emergence-aware processing with meta-cognition"""
        try:
            # Initial retrieval
            context_memories = self.retrieve(query, k=4)
            context = "\n".join([mem['content'] for mem in context_memories])
            
            # Emergence detection
            if self._requires_emergence(query, context_memories):
                return self._emergence_cascade(query, context, depth=1)
            else:
                prompt = f"Relevant Context:\n{context}\n\nQuery: {query}\nResponse:"
                return self._generate_response(prompt)
                
        except Exception as e:
            logging.error(f"Query processing failed: {e}")
            return "I encountered an error processing your request"

    def _generate_response(self, prompt, max_tokens=512):
        """Generate response using quantized LLM"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm(
                self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                ),
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            return response[0]['generated_text'].split("Response:")[-1].strip()
        except Exception as e:
            logging.error(f"Response generation failed: {e}")
            return "I couldn't generate a response"

    def _requires_emergence(self, query, memories):
        """Determine if advanced processing is needed"""
        if not memories:
            return True
            
        try:
            # Calculate composite relevance score
            relevance_score = np.mean([mem['weight'] for mem in memories])
            complexity_score = len(query.split()) / 20.0
            return (relevance_score * complexity_score) < self.emergence_threshold
        except:
            return True

    def _emergence_cascade(self, query, context, depth):
        """Recursive emergence processing with depth control"""
        if depth > 3:  # Safety limit
            return "I've reached my reasoning depth limit for this query"
            
        try:
            # Generate reasoning chain
            prompt = (
                f"## Deep Reasoning Task (Level {depth})\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n"
                "Reason step-by-step considering multiple perspectives:\n1."
            )
            reasoning = self._generate_response(prompt, max_tokens=1024)
            
            # Create intermediate memory
            mem_id = self.add_memory(
                content=f"Reasoning Chain: {reasoning}",
                context=context
            )
            
            # Retrieve again with new context
            new_context_memories = self.retrieve(query, k=5)
            new_context = "\n".join([mem['content'] for mem in new_context_memories])
            
            # Check if resolved
            if not self._requires_emergence(query, new_context_memories):
                response_prompt = (
                    f"## Final Response Synthesis\n"
                    f"Reasoning Process:\n{reasoning}\n\n"
                    f"Augmented Context:\n{new_context}\n\n"
                    f"Query: {query}\nFinal Answer:"
                )
                return self._generate_response(response_prompt)
            
            # Recursive emergence
            return self._emergence_cascade(query, new_context, depth+1)
            
        except Exception as e:
            logging.error(f"Emergence processing failed at depth {depth}: {e}")
            return "I encountered an error during deep reasoning"

    def get_memory(self, mem_id):
        try:
            cur = self.memory_db.cursor()
            cur.execute('''SELECT * FROM memories WHERE id = ?''', (mem_id,))
            row = cur.fetchone()
            
            if not row:
                return None
                
            return {
                'id': row[0],
                'content': row[1],
                'tags': row[2].split(',') if row[2] else [],
                'weight': row[3],
                'last_accessed': row[4],
                'links': [int(x) for x in row[5].split(',')] if row[5] else [],
                'access_count': row[6],
                'semantic_density': row[7]
            }
        except Exception as e:
            logging.error(f"Memory retrieval failed: {e}")
            return None

    def consolidate_memories(self):
        """Adaptive memory consolidation with semantic clustering"""
        try:
            # Get all memories
            cur = self.memory_db.cursor()
            cur.execute('''SELECT id, content, weight FROM memories''')
            memories = cur.fetchall()
            
            if len(memories) < 2:
                return
                
            # Cluster similar memories
            clusters = self._semantic_clustering(memories)
            
            # Merge cluster members
            for cluster in clusters:
                if len(cluster) > 1:
                    self._merge_cluster(cluster)
                    
            logging.info(f"Consolidated {len(clusters)} memory clusters")
            
        except Exception as e:
            logging.error(f"Consolidation failed: {e}")

    def _semantic_clustering(self, memories, threshold=0.88):
        """Hierarchical memory clustering"""
        try:
            # Get all embeddings from database
            embeddings = []
            valid_ids = []
            
            for mem_id, _, _ in memories:
                mem = self.get_memory(mem_id)
                if mem:
                    cur = self.memory_db.cursor()
                    cur.execute("SELECT embedding FROM memories WHERE id = ?", (mem_id,))
                    result = cur.fetchone()
                    if result and result[0]:
                        embedding = np.frombuffer(result[0], dtype=np.float32)
                        embeddings.append(embedding)
                        valid_ids.append(mem_id)
            
            if len(embeddings) < 2:
                return []
                
            embeddings = np.vstack(embeddings)
            
            # Build clustering index
            cluster_index = faiss.IndexFlatIP(self.dimension)
            cluster_index.add(embeddings)
            
            # Find neighbors
            _, I = cluster_index.search(embeddings, 5)
            
            # Form clusters
            clusters = []
            visited = set()
            
            for idx, mem_id in enumerate(valid_ids):
                if mem_id in visited:
                    continue
                    
                cluster = {mem_id}
                queue = [mem_id]
                visited.add(mem_id)
                
                while queue:
                    current_id = queue.pop(0)
                    current_idx = valid_ids.index(current_id)
                    
                    for neighbor_idx in I[current_idx]:
                        if neighbor_idx >= len(valid_ids):
                            continue
                            
                        neighbor_id = valid_ids[neighbor_idx]
                        if neighbor_id in visited:
                            continue
                            
                        # Check similarity
                        sim = np.dot(
                            embeddings[current_idx], 
                            embeddings[neighbor_idx]
                        )
                        if sim > threshold:
                            cluster.add(neighbor_id)
                            queue.append(neighbor_id)
                            visited.add(neighbor_id)
                
                if len(cluster) > 1:
                    clusters.append(list(cluster))
                    
            return clusters
            
        except Exception as e:
            logging.error(f"Clustering failed: {e}")
            return []

    def _merge_cluster(self, cluster_ids):
        """Merge a cluster of related memories"""
        try:
            # Retrieve cluster memories
            memories = [self.get_memory(mid) for mid in cluster_ids]
            valid_memories = [m for m in memories if m is not None]
            
            if len(valid_memories) < 2:
                return
                
            # Find core memory (highest weight)
            core_memory = max(valid_memories, key=lambda m: m['weight'])
            related_contents = [
                m['content'] for m in valid_memories 
                if m['id'] != core_memory['id']
            ]
            
            # Create merged content
            merged_content = (
                f"Core Memory: {core_memory['content']}\n\n"
                f"Related Aspects:\n- " + "\n- ".join(related_contents)
            
            # Create new memory
            new_id = self.add_memory(merged_content)
            
            # Update weights (average with consolidation bonus)
            avg_weight = np.mean([m['weight'] for m in valid_memories])
            new_weight = min(1.0, avg_weight * 1.2)
            
            cur = self.memory_db.cursor()
            cur.execute('''UPDATE memories SET weight = ? WHERE id = ?''', 
                       (new_weight, new_id))
            
            # Remove original memories
            self.index.remove_ids(np.array(cluster_ids, dtype=np.int64))
            placeholders = ','.join('?' * len(cluster_ids))
            cur.execute(f'''DELETE FROM memories WHERE id IN ({placeholders})''', cluster_ids)
            self.memory_db.commit()
            
        except Exception as e:
            logging.error(f"Cluster merging failed: {e}")

    def save_state(self):
        """Persist memory system state to disk"""
        if not self.persist:
            return
            
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{self.db_path}.index")
            logging.info("Memory state persisted successfully")
        except Exception as e:
            logging.error(f"State persistence failed: {e}")

    def close(self):
        """Clean up resources"""
        self.save_state()
        self.memory_db.close()
        logging.info("Memory system shut down cleanly")
