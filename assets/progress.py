!pip install pactools

from pactools import simulate_pac
from pactools.grid_search import ExtractDriver, AddDriverDelay
from pactools.grid_search import DARSklearn, MultipleArray
from pactools.grid_search import GridSearchCVProgressBar

fs = 200.  # Hz
high_fq = 50.0  # Hz
low_fq = 5.0  # Hz
low_fq_width = 1.0  # Hz

n_epochs = 3
n_points = 10000
noise_level = 0.4

low_sig = np.array([
    simulate_pac(n_points=n_points, fs=fs, high_fq=high_fq, low_fq=low_fq,
                 low_fq_width=low_fq_width, noise_level=noise_level,
                 random_state=i) for i in range(n_epochs)
])

X = MultipleArray(low_sig, None)