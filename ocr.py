import numpy as np
import json
import os


class OCRNeuralNetwork:
    def __init__(self, hidden_nodes=32, learning_rate=0.03, weights_file="weights.json"):
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate
        self.weights_file = weights_file

        self.theta1 = np.random.randn(hidden_nodes, 400) * np.sqrt(2.0 / 400)
        self.theta2 = np.random.randn(10, hidden_nodes) * np.sqrt(2.0 / hidden_nodes)

        self.b1 = np.zeros((hidden_nodes, 1))
        self.b2 = np.zeros((10, 1))

        self.load()

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def softmax(self, z):
        z = z - np.max(z)
        e = np.exp(z)
        return e / np.sum(e, axis=0)

    def forward(self, x):
        x = x.reshape(400, 1)

        z1 = self.theta1 @ x + self.b1
        a1 = self.sigmoid(z1)

        z2 = self.theta2 @ a1 + self.b2
        a2 = self.softmax(z2)

        return x, a1, a2

    def train_sample(self, image, label):
        x, a1, a2 = self.forward(np.array(image))

        t = np.zeros((10, 1))
        t[label] = 1.0

        d2 = a2 - t
        d1 = (self.theta2.T @ d2) * (a1 * (1 - a1))

        self.theta2 -= self.learning_rate * (d2 @ a1.T)
        self.theta1 -= self.learning_rate * (d1 @ x.T)
        self.b2 -= self.learning_rate * d2
        self.b1 -= self.learning_rate * d1

    def train_batch(self, batch, epochs=5):
        batch = list(batch)

        for _ in range(epochs):
            np.random.shuffle(batch)
            for item in batch:
                self.train_sample(item["y0"], int(item["label"]))

    def predict(self, image):
        _, _, a2 = self.forward(np.array(image))
        return int(np.argmax(a2))

    def save(self):
        data = {
            "theta1": self.theta1.tolist(),
            "theta2": self.theta2.tolist(),
            "b1": self.b1.tolist(),
            "b2": self.b2.tolist(),
            "hidden_nodes": self.hidden_nodes
        }
        with open(self.weights_file, "w") as f:
            json.dump(data, f)

    def load(self):
        if not os.path.exists(self.weights_file):
            return
        try:
            with open(self.weights_file) as f:
                data = json.load(f)
            self.theta1 = np.array(data["theta1"])
            self.theta2 = np.array(data["theta2"])
            self.b1 = np.array(data["b1"])
            self.b2 = np.array(data["b2"])
        except Exception:
            pass
