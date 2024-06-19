import numpy as np
from river import datasets


class SineClusters(datasets.base.SyntheticDataset):
    def __init__(
        self,
        drift_frequency,
        stream_length,
        seed: int or None = None,
    ):
        super().__init__(
            task=datasets.base.MULTI_CLF,
            n_features=4,
        )
        self.drift_frequency = drift_frequency
        self.stream_length = stream_length
        self.centroids = None
        self.sine_period = 500
        self.indices = np.linspace(0, 2 * np.pi, self.sine_period + 1)
        self.seed = seed
        self.rng = None
        self.get_features = None
        self.drifts = [i * self.drift_frequency for i in range(int(stream_length / drift_frequency))][1:]

    def get_label(self, features):
        features = features[:2]
        distances = np.linalg.norm(self.centroids - features, axis=1),
        closest_centroid = np.argmin(distances)
        return closest_centroid % 3

    def set_centroids(self):
        centroids = []
        for i in range(6):
            centroids.append(self.rng.uniform(low=-1, high=1, size=2))
        self.centroids = np.array(centroids)

    def __iter__(self):
        self.rng = np.random.default_rng(self.seed)
        i = 0
        while i < self.stream_length:
            if i % self.drift_frequency == 0:
                self.drift()
            x = self.get_features(i)
            x += self.rng.normal(0, 0.25, 4)
            y = self.get_label(x)
            x = {i: x_component for i, x_component in enumerate(x)}
            i += 1
            yield x, y

    def drift(self):
        self.set_centroids()
        new_concept = self.rng.choice([self.concept_one, self.concept_two, self.concept_three], 1)[0]
        while new_concept == self.get_features:
            new_concept = self.rng.choice([self.concept_one, self.concept_two, self.concept_three], 1)[0]

        print(new_concept)
        self.get_features = new_concept

    def concept_one(self, i):
        """
        Sine, Cosine, Sine and Cosine
        """
        x_sin = np.sin(self.indices[i % self.sine_period])
        x_cos = np.cos(self.indices[i % self.sine_period])
        x = [
            x_sin,
            x_cos,
            x_sin,
            x_cos,
        ]
        return x

    def concept_two(self, i):
        """
        -Sine, Cosine, Sine and Cosine
        """
        x_sin = np.sin(self.indices[i % self.sine_period])
        x_cos = np.cos(self.indices[i % self.sine_period])
        x = [
            -x_sin,
            x_cos,
            x_sin,
            x_cos,
        ]
        return x

    def concept_three(self, i):
        """
        -Cosine, -Cosine, Sine and Cosine
        """
        x_sin = np.sin(self.indices[i % self.sine_period])
        x_cos = np.cos(self.indices[i % self.sine_period])
        x = [
            -x_cos,
            -x_cos,
            x_sin,
            x_cos,
        ]
        return x
