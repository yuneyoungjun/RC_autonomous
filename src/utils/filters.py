# utils/filters.py

import numpy as np
from collections import deque

class FilterWithQueue:
    def __init__(self, queue_len=5, alpha=0.2, threshold=30):
        self.queue_len = queue_len
        self.alpha = alpha
        self.threshold = threshold

        self.queue = deque(maxlen=self.queue_len)
        self.filtered_value = 0.0

    def iir_filter(self, new_val):
        self.filtered_value = self.alpha * new_val + (1 - self.alpha) * self.filtered_value
        return self.filtered_value

    def is_outlier(self, new_val):
        if len(self.queue) < self.queue_len:
            return False
        mean = np.mean(self.queue)
        std = np.std(self.queue)
        return abs(new_val - mean) > max(self.threshold, 2 * std)

    def update(self, new_val):
        if not self.is_outlier(new_val):
            self.queue.append(new_val)
            return self.iir_filter(new_val), False  # False: not outlier
        return self.filtered_value, True  # True: is outlier

    def reset(self):
        self.queue.clear()
        self.filtered_value = 0.0
