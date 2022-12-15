from typing import Tuple, Iterator, Union
import math
from pathlib import Path
from copy import deepcopy
import random
from skimage.transform import resize

import numpy as np
import cv2

from dataset.fma_dataset_reader import FmaDatasetReader
from audio import read_audio_file, stft, get_stft_window_func_by_name
from phase_ops import instantaneous_frequency


def save_spectogram_as_fig(spectogram: np.ndarray,
                           output_path: Union[Path, str],
                           scale: float = 1):
    output_path = Path(output_path)
    spectogram[spectogram < -1] = -1
    spectogram[spectogram > 1] = 1
    spectogram = ((spectogram + 1) * 0.5 * 255).astype(np.uint8)
    zeros = np.zeros((spectogram.shape[2], spectogram.shape[1]))
    if spectogram.shape[0] > 2:
        left_path = Path(str(output_path) + '_left.png')
        right_path = Path(str(output_path) + '_right.png')
        left_image = np.stack([zeros,
                               spectogram[1, :, :].T,
                               spectogram[0, :, :].T], axis=2)
        right_image = np.stack([zeros,
                                spectogram[3, :, :].T,
                                spectogram[2, :, :].T], axis=2)
        if scale != 1:
            left_image = cv2.resize(left_image, None, None, scale, scale)
            right_image = cv2.resize(right_image, None, None, scale, scale)
        cv2.imwrite(str(left_path), left_image)
        cv2.imwrite(str(right_path), right_image)
    else:
        if output_path.suffix != '.png':
            output_path = Path(str(output_path) + '.png')
        image = np.stack([zeros,
                          spectogram[1, :, :].T,
                          spectogram[0, :, :].T], axis=2)
        if scale != 1:
            image = cv2.resize(image, None, None, scale, scale)
        cv2.imwrite(str(output_path), image)


class MlAudioDataset:
    def __init__(self,
                 reader: FmaDatasetReader,
                 config: dict):
        self.model_config = deepcopy(config['gansynth'])
        cfg_blocks = self.model_config['discriminator']['blocks']

        self.in_shape = (int(self.model_config['in_shape'][0]),
                         int(self.model_config['in_shape'][1]))
        in_shape = self.in_shape
        self.level_shapes = [in_shape]
        for cfg_block in cfg_blocks:
            stride = int(cfg_block['stride'])
            in_shape = (math.ceil(in_shape[0] / stride), math.ceil(in_shape[1] / stride))
            self.level_shapes.append(in_shape)

        self.reader = reader
        self.stft_config = deepcopy(config['stft'])
        self.genre_vector_size = len(self.reader.genres)
        self.stft_window_func = get_stft_window_func_by_name(self.stft_config['window_func'])
        self.samplerate = int(self.stft_config['samplerate'])
        self.use_instantaneous_frequency = bool(self.stft_config['use_instantaneous_frequency'])

    def get_samplerate(self, level_index: int = 0) -> int:
        return int(self.samplerate // (2 ** level_index))

    def __len__(self) -> int:
        return len(self.reader)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return (self.get_item(item_index) for item_index in range(len(self.reader)))

    def iter_level(self, level_index: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return (self.get_item(item_index, level_index) for item_index in range(len(self.reader)))

    def get_item(self, index: int, level_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        track = self.reader.get_track(index)

        samplerate = self.get_samplerate(level_index)

        spectogram_path = track['filepath'].parent / \
                          (track['filepath'].stem + f'_{samplerate}.npy')
        if spectogram_path.exists():
            spectogram = np.load(str(spectogram_path), allow_pickle=True)
        else:
            signal_samplerate, signal = read_audio_file(track['filepath'], samplerate)
            assert signal_samplerate == samplerate
            if len(signal.shape) == 1:
                signal = signal.reshape(signal.shape[0], 1)
            # if signal.shape[1] == 1:
            #     signal = np.concatenate([signal, signal], axis=1)
            assert len(signal.shape) == 2 and (signal.shape[1] == 2 or signal.shape[1] == 1), \
                f'Invalid signal shape: {signal.shape}'
            spectogram, source_slice = stft(signal,
                                            int(self.stft_config['frame_size']),
                                            float(self.stft_config['overlap_fac']),
                                            self.stft_window_func)
            if len(spectogram.shape) == 2:
                spectogram.reshape(1, *spectogram.shape)
            spectogram = np.stack([np.abs(spectogram[0, :, :]),
                                   np.angle(spectogram[0, :, :])], axis=0)
            np.save(str(spectogram_path), spectogram, allow_pickle=True)
        genre_vector = np.zeros((self.genre_vector_size,), dtype=float)
        for genre in track['genres']:
            genre_vector[genre['out_id']] = 1

        level_shape = self.level_shapes[level_index]

        if spectogram.shape[1] > level_shape[0]:
            rand_offset = random.randint(0, spectogram.shape[1] - level_shape[0])
            spectogram = spectogram[:, rand_offset:rand_offset + level_shape[0], :]
        elif spectogram.shape[1] < level_shape[0]:
            extra_part = np.zeros((spectogram.shape[0],
                                   level_shape[0] - spectogram.shape[1],
                                   spectogram.shape[2]),
                                  dtype=spectogram.dtype)
            spectogram = np.append(spectogram, extra_part, axis=1)

        if spectogram.shape[2] != level_shape[1]:
            spectogram = np.stack([resize(s, level_shape) for s in spectogram], axis=0)
        if self.use_instantaneous_frequency:
            spectogram = instantaneous_frequency(spectogram)

        return spectogram, genre_vector


class MlAudioBatchCollector:
    def __init__(self, dataset: MlAudioDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = list(range(len(self.dataset)))

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return (self.get_batch(item_index) for item_index in range(len(self)))

    def iter_level(self, level_index: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return (self.get_batch(item_index, level_index) for item_index in range(len(self)))

    def get_batch(self, batch_index: int, level_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        offset = batch_index * self.batch_size
        batch = [self.dataset.get_item(self.indices[item_index], level_index)
                 for item_index in range(offset, offset + self.batch_size)]
        spectograms = np.stack([b[0] for b in batch], axis=0)
        genre_vectors = np.stack([b[1] for b in batch], axis=0)
        return spectograms, genre_vectors
