from typing import Tuple, Union, Dict, Iterable

import sys
import numpy as np
from tqdm import tqdm
from dataset import MlAudioDataset


class DataNormalizer:

    @staticmethod
    def compute_ranges(dataset: Union[MlAudioDataset, Iterable[Tuple[np.ndarray, np.ndarray]]]) \
            -> Dict[str, Tuple[float, float]]:
        min_amplitude, max_amplitude = sys.float_info.max, -sys.float_info.max
        min_phase, max_phase = sys.float_info.max, -sys.float_info.max
        for spectogram, genre_vector in tqdm(dataset, 'Computing'):
            if len(spectogram.shape) == 3:
                s_a_min = spectogram[0, :, :].min()
                s_a_max = spectogram[0, :, :].max()
                s_p_min = spectogram[1, :, :].min()
                s_p_max = spectogram[1, :, :].max()
                if spectogram.shape[0] > 2:
                    s_a_min = min(spectogram[0, :, :].min(), spectogram[2, :, :].min())
                    s_a_max = max(spectogram[0, :, :].max(), spectogram[2, :, :].max())
                    s_p_min = min(spectogram[1, :, :].min(), spectogram[3, :, :].min())
                    s_p_max = max(spectogram[1, :, :].max(), spectogram[3, :, :].max())
            else:
                s_a_min = spectogram[:, 0, :, :].min()
                s_a_max = spectogram[:, 0, :, :].max()
                s_p_min = spectogram[:, 1, :, :].min()
                s_p_max = spectogram[:, 1, :, :].max()
                if spectogram.shape[1] > 2:
                    s_a_min = min(spectogram[:, 0, :, :].min(), spectogram[:, 2, :, :].min())
                    s_a_max = max(spectogram[:, 0, :, :].max(), spectogram[:, 2, :, :].max())
                    s_p_min = min(spectogram[:, 1, :, :].min(), spectogram[:, 3, :, :].min())
                    s_p_max = max(spectogram[:, 1, :, :].max(), spectogram[:, 3, :, :].max())

            if s_a_min < min_amplitude:
                min_amplitude = s_a_min
            if s_a_max > max_amplitude:
                max_amplitude = s_a_max
            if s_p_min < min_phase:
                min_phase = s_p_min
            if s_p_max > max_phase:
                max_phase = s_p_max
        return {
            'amplitude': (min_amplitude, max_amplitude),
            'phase': (min_phase, max_phase)
        }

    def __init__(self,
                 amplitude_range: Tuple[float, float],
                 phase_range: Tuple[float, float]):
        self.amplitude_range = (float(amplitude_range[0]), float(amplitude_range[1]))
        self.phase_range = (float(phase_range[0]), float(phase_range[1]))
        self.amplitude_coeffs = (2 / (amplitude_range[1] - amplitude_range[0]),
                                 - (2 * amplitude_range[0]) / (amplitude_range[1] - amplitude_range[0]) - 1)
        self.phase_coeffs = (2 / (phase_range[1] - phase_range[0]),
                             - (2 * phase_range[0]) / (phase_range[1] - phase_range[0]) - 1)

    def normalize(self, spectogram: np.ndarray, copy: bool = False) -> np.ndarray:
        if copy:
            spectogram = spectogram.copy()
        if len(spectogram.shape) == 3:
            spectogram[0, :, :] = spectogram[0, :, :] * self.amplitude_coeffs[0] + self.amplitude_coeffs[1]
            spectogram[1, :, :] = spectogram[1, :, :] * self.phase_coeffs[0] + self.phase_coeffs[1]
            if spectogram.shape[0] > 2:
                spectogram[2, :, :] = spectogram[2, :, :] * self.amplitude_coeffs[0] + self.amplitude_coeffs[1]
                spectogram[3, :, :] = spectogram[3, :, :] * self.phase_coeffs[0] + self.phase_coeffs[1]
        else:
            spectogram[:, 0, :, :] = spectogram[:, 0, :, :] * self.amplitude_coeffs[0] + self.amplitude_coeffs[1]
            spectogram[:, 1, :, :] = spectogram[:, 1, :, :] * self.phase_coeffs[0] + self.phase_coeffs[1]
            if spectogram.shape[1] > 2:
                spectogram[:, 2, :, :] = spectogram[:, 2, :, :] * self.amplitude_coeffs[0] + self.amplitude_coeffs[1]
                spectogram[:, 3, :, :] = spectogram[:, 3, :, :] * self.phase_coeffs[0] + self.phase_coeffs[1]
        spectogram[spectogram < -1] = -1
        spectogram[spectogram > 1] = 1
        return spectogram

    def unnormalize(self, spectogram: np.ndarray, copy: bool = False) -> np.ndarray:
        if copy:
            spectogram = spectogram.copy()
        spectogram[spectogram < -1] = -1
        spectogram[spectogram > 1] = 1
        if len(spectogram.shape) == 3:
            spectogram[0, :, :] = (spectogram[0, :, :] - self.amplitude_coeffs[1]) * \
                                  (1 / self.amplitude_coeffs[0])
            spectogram[1, :, :] = (spectogram[1, :, :] - self.phase_coeffs[1]) * \
                                  (1 / self.phase_coeffs[0])
            if spectogram.shape[0] > 2:
                spectogram[2, :, :] = (spectogram[2, :, :] - self.amplitude_coeffs[1]) * \
                                      (1 / self.amplitude_coeffs[0])
                spectogram[3, :, :] = (spectogram[3, :, :] - self.phase_coeffs[1]) * \
                                      (1 / self.phase_coeffs[0])
        else:
            spectogram[:, 0, :, :] = (spectogram[:, 0, :, :] - self.amplitude_coeffs[1]) * \
                                     (1 / self.amplitude_coeffs[0])
            spectogram[:, 1, :, :] = (spectogram[:, 1, :, :] - self.phase_coeffs[1]) * \
                                     (1 / self.phase_coeffs[0])
            if spectogram.shape[1] > 2:
                spectogram[:, 2, :, :] = (spectogram[:, 2, :, :] - self.amplitude_coeffs[1]) * \
                                         (1 / self.amplitude_coeffs[0])
                spectogram[:, 3, :, :] = (spectogram[:, 3, :, :] - self.phase_coeffs[1]) * \
                                         (1 / self.phase_coeffs[0])
        return spectogram


if __name__ == '__main__':
    from pathlib import Path
    import yaml
    from dataset import FmaDatasetReader, MlAudioBatchCollector
    import os

    SCRIPT_PATH = Path(os.path.abspath(__file__)).parent

    config = yaml.safe_load(open(SCRIPT_PATH / '..' / 'data' / 'data.yaml', 'r'))
    reader = FmaDatasetReader(config)
    ml_audio_dataset = MlAudioDataset(reader, config)
    batch_collector = MlAudioBatchCollector(ml_audio_dataset, 16)

    ranges = DataNormalizer.compute_ranges(ml_audio_dataset)
    print(f'Ranges: {ranges}')

    data_normalizer = DataNormalizer(config['datasets']['ranges']['amplitude'],
                                     config['datasets']['ranges']['phase'])
    ranges2 = DataNormalizer.compute_ranges(((data_normalizer.normalize(s), g_v) for s, g_v in batch_collector))
    print(f'Ranges 2: {ranges2}')
