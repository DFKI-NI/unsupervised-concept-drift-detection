import numpy as np
from river.datasets.synth.waveform import Waveform


class WaveformDrift2(Waveform):
    def __init__(
        self,
        drift_frequency: int,
        stream_length: int,
        seed: int or None = None,
        has_noise: bool = False,
    ):
        """
        Init a synthetic data stream based on River's Waveform data stream

        :param drift_frequency: the interval at which concept drift happens
        :param stream_length: the length of the data stream
        :seed: the seed
        :has_noise: whether to include noisy features or not
        """
        super().__init__(seed, has_noise)
        self.h_functions = [
            self._H_FUNCTION + 0,
            self._H_FUNCTION + 6,
            6 - self._H_FUNCTION,
            self._H_FUNCTION * -1,
        ]
        self._H_FUNCTION = self.h_functions[0]
        self.drift_frequency = drift_frequency
        self.stream_length = stream_length
        self.rng = None
        self.drifts = [i * self.drift_frequency for i in range(int(stream_length / drift_frequency))][1:]

    def drift(self):
        """
        Change the waveform functions.
        """
        new_function = self.rng.choice(self.h_functions, 1)[0]
        if np.all(self._H_FUNCTION == new_function):
            return self.drift()
        else:
            return new_function

    def __iter__(self):
        self._H_FUNCTION = self.h_functions[0]
        self.rng = np.random.default_rng(seed=self.seed)
        i = 0
        for x, y in super().__iter__():
            if i == self.stream_length:
                break
            if i % self.drift_frequency == 0:
                self._H_FUNCTION = self.drift()
            i += 1
            yield x, y
