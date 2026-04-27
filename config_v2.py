from datasets import (
    InsectsAbruptBalanced,
    InsectsGradualBalancedRaw,
    InsectsIncrementalBalancedRaw,
    InsectsIncrementalAbruptBalancedRaw,
    InsectsIncrementalReoccurringBalancedRaw,
    OzoneLevelDetection,
    ElectricityNormalized,
    PokerHandRaw,
    SineClusters,
    WaveformDrift2,
)
from detectors import *
from optimization.model_optimizer import ModelOptimizer
from optimization.parameter import Parameter


class Configuration:
    streams = [
        #ElectricityNormalized(),
        #PokerHandRaw(),
        #SineClusters(drift_frequency=5000, stream_length=50000),
        #WaveformDrift2(drift_frequency=5000, stream_length=50000),
        OzoneLevelDetection(),
    ]
    n_training_samples = 1000
    models = [
        ModelOptimizer(
            base_model=DiscriminativeDriftDetector2019,
            parameters=[
                Parameter("n_reference_samples", values=[250, 500]),
                Parameter("recent_samples_proportion", values=[1.0]),
                Parameter("threshold", values=[0.7, 0.8]),
            ],
            seeds=None,
            n_runs=1,
        ),
    ]
