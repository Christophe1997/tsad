import numpy as np


class FixSizedQueue:

    def __init__(self, size):
        self.capacity = size
        self.__data = []
        self.size = 0

    def enqueue(self, val):
        self.__data.append(val)
        self.size += 1
        if self.size > self.capacity:
            self.__data.pop(0)
            self.size -= 1

    def dequeue(self):
        self.size -= 1
        return self.__data.pop(0)

    def to_numpy(self):
        return np.array(self.__data)


class Detector:

    def __init__(self, model, history_w, predict_w, init_threshold=2):
        self.model = model
        self.queue = FixSizedQueue(history_w + predict_w - 1)
        self.threshold = init_threshold
        self.cache = FixSizedQueue(predict_w)
        self.history_deviation = []

    def init(self, data):
        for e in data:
            self.queue.enqueue(e)

    def detect(self, y):
        pass
