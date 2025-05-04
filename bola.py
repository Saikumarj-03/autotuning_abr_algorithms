import numpy as np

class BOLA:
    def __init__(self):
        self.bitrates = [300, 750, 1200, 1850, 2850, 4300]  # kbps
        self.utilities = np.log(np.array(self.bitrates) / self.bitrates[0])
        self.chunk_duration = 4.0  # seconds

    def run(self, buffer_size, gamma):
        scores = []
        for i in range(len(self.bitrates)):
            score = self.utilities[i] + gamma / buffer_size
            scores.append(score)
        return int(np.argmax(scores))
