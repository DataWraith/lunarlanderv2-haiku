class AvgMeter:
    """Efficiently keeps track of the average of the last `size` values passed in"""

    def __init__(self, size):
        self.size = size
        self.store = [0.0] * size
        self.sum = 0.0
        self.index = 0

    def append(self, x):
        idx = self.index % self.size

        self.sum -= self.store[idx]
        self.store[idx] = x
        self.sum += self.store[idx]

        self.index += 1

    def mean(self):
        return self.sum / max(1, min(self.size, self.index))
