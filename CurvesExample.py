class MemoryUsageCurve:
    """Determines optimal memories per query using Ebbinghaus forgetting curve"""
    def __init__(self):
        self.k_min = 3
        self.k_max = 15
        self.decay_rate = 0.92
        self.retention_factor = 1.8
        
    def optimal_k(self, t):
        """
        k(t) = k_min + (k_max - k_min) * (1 - e^{-γt})
        Where γ = retention_factor / (1 + log(t+1))
        """
        γ = self.retention_factor / (1 + math.log(t + 1))
        return min(self.k_max, 
                  max(self.k_min, 
                      round(self.k_min + (self.k_max - self.k_min) * (1 - math.exp(-γ * t)))
                  )
               )
    
    def update(self, actual_k, accuracy):
        """Adapt curve based on performance"""
        # Adjust retention factor based on accuracy
        if accuracy > 0.85:
            self.retention_factor *= 1.05
        else:
            self.retention_factor *= 0.95
            
        # Ensure within bounds
        self.retention_factor = max(1.2, min(2.5, self.retention_factor))

class BatchLearningCurve:
    """Adaptive batching using Wright's Law"""
    def __init__(self):
        self.min_batch = 4
        self.max_batch = 64
        self.learning_rate = 0.15
        self.complexity_factor = 1.0
        
    def optimal_batch_size(self, t, available_items, avg_complexity):
        """
        B(t,c) = min(B_max, max(B_min, round(β * t^α / c)))
        Where:
          β = base learning rate
          α = learning exponent (0.3)
          c = complexity factor
        """
        α = 0.3
        base_size = self.learning_rate * (t ** α) / (avg_complexity * self.complexity_factor)
        bounded_size = min(self.max_batch, 
                          max(self.min_batch, 
                              round(base_size))
        return min(bounded_size, available_items)
    
    def update(self, used_batch, available_items, processing_time):
        """Update learning parameters"""
        # Calculate efficiency: items processed per second
        efficiency = used_batch / max(0.001, processing_time)
        
        # Adjust learning rate based on efficiency
        if efficiency > (used_batch / self.expected_time(used_batch)):
            self.learning_rate *= 1.05
        else:
            self.learning_rate *= 0.97
            
        # Adjust complexity factor
        self.complexity_factor = max(0.7, min(1.3, self.complexity_factor))
        
    def expected_time(self, batch_size):
        """Expected processing time model"""
        return 0.01 * batch_size + 0.0001 * (batch_size ** 1.5)
