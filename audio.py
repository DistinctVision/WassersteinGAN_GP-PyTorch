# -*- coding: utf-8 -*-
import math
from typing import Callable, Union, Tuple, Optional
from pathlib import Path
import time

import numpy as np
from numpy.lib import stride_tricks

import scipy.io.wavfile as wavfile
from matplotlib import pyplot as plt

from pydub import AudioSegment


def get_stft_window_func_by_name(name: str) -> Optional[Callable[[int], np.ndarray]]:
    return {
        'none': None,
        'hanning': np.hanning,
        'hamming': np.hamming
    }[name]


def stft(signal: np.ndarray,
         frame_size: int,
         overlap_fac: float = 0.33,
         window_func: Optional[Callable[[int], np.ndarray]] = np.hanning
         ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    The stft function takes a signal and returns the short time fourier transform of that signal.

    Note:
        It does this by taking the frame size, hop size, and window function as parameters.
        The frame size is how many samples are in each fft calculation (frame_size = 1024).
        The hop size is how many samples are skipped between each fft calculation (hop_size=512).
        The window function is used to calculate each FFT calculation on a smaller sample of the
        original audio file. This helps with noise reduction because only a small section of audio
        data is being analyzed at one time.
    
    Args:
        signal: Input signal of the meidia file
        frame_size: The size of the window used to split up the signal
        overlap_fac: The overlap between frames
        window_func: Specify the window function
    
    Returns:
        A tuple of two elements: spectrums, source_slice
    """

    assert len(signal.shape) <= 2, "Invalid the shape of the signal array"

    if len(signal.shape) < 2:
        signal = signal.reshape((-1, 1))

    hop_size = int(frame_size - math.floor(overlap_fac * frame_size))

    source_slice = (frame_size // 2, signal.shape[0] + frame_size // 2)

    out_n_frames = math.ceil((signal.shape[0] + (frame_size // 2) * 2) / float(hop_size))

    spectrums = []

    for channel_signal in signal.T:
        # zeros at begin and at end (thus samples can be fully covered by frames)
        channel_signal = np.concatenate([np.zeros((frame_size // 2,), dtype=signal.dtype),
                                         channel_signal,
                                         np.zeros(out_n_frames * hop_size -
                                                  source_slice[1] + frame_size,
                                         dtype=signal.dtype)])

        # stride_tricks.sliding_window_view is not a suitable function for this case
        frames = stride_tricks.as_strided(channel_signal,
                                          shape=(out_n_frames, frame_size),
                                          strides=(channel_signal.strides[0] * hop_size,
                                                   channel_signal.strides[0])).copy()

        channel_spectrums = np.fft.rfft(frames)  # Do we need to specify the parameter "n"?

        spectrums.append(channel_spectrums)
    spectrums = np.stack(spectrums, axis=0)

    if window_func is not None:
        window_kernel = window_func(spectrums.shape[2]) + 1e-2
        window_kernel /= window_kernel.sum()
        spectrums.real *= window_kernel

    return spectrums, source_slice


def istft(spectrums: np.ndarray,
          source_slice: Optional[Tuple[int, int]],
          overlap_fac: float = 0.33,
          window_func: Optional[Callable[[int], np.ndarray]] = np.hanning
          ) -> np.ndarray:
    """
    The istft function takes a spectrogram and converts it back into an audio signal.

    Note:
        The spectrogram is expected to be in the form of numpy array, dimension:
        - The number of channels (1 mono, 2 stereo)
        - The number of frames (the time axis)
        - The frequency bins for that frame. (This should be equal to the window size used
          when computing the STFT.)

    Args:
        spectrums: The spectrums of the audio signal
        source_slice: Cut the signal to the correct length
        overlap_fac: The number of samples that are overlapped between adjacent frames
        window_func: Specify the window function

    Returns:
        An audio signal
    """
    assert 1 < len(spectrums.shape) <= 3, "Invalid the shape of the spectrums array"

    if len(spectrums.shape) < 3:
        spectrums = spectrums.reshape((-1, spectrums.shape[0], spectrums.shape[1]))

    frame_size = (spectrums.shape[2] - spectrums.shape[2] % 2) * 2

    if window_func is not None:
        window_kernel = window_func(spectrums.shape[2]) + 1e-2
        window_kernel /= window_kernel.sum()
        spectrums.real /= window_kernel

    hop_size = int(frame_size - math.floor(overlap_fac * frame_size))

    signal = []
    for channel_spectrums in spectrums:
        frames = np.fft.irfft(channel_spectrums)
        if frames.shape[1] < frame_size:
            frames = np.concatenate([frames,
                                     np.zeros((frames.shape[0], frame_size - frames.shape[1]),
                                              dtype=frames.dtype)],
                                    axis=1)
        channel_signal = np.zeros((frames.shape[0] * hop_size + frame_size), dtype=frames.dtype)
        counter = np.zeros(channel_signal.shape, dtype=int)

        for frame_index, frame in enumerate(frames):
            frame_offset = frame_index * hop_size
            channel_signal[frame_offset:frame_offset+frame_size] += frame
            counter[frame_offset:frame_offset+frame_size] += 1

        not_null_signal_mask = counter > 0
        channel_signal[np.logical_not(not_null_signal_mask)] = 0.0
        channel_signal[not_null_signal_mask] /= counter[not_null_signal_mask]
        if source_slice is not None:
            channel_signal = channel_signal[source_slice[0]:source_slice[1]]
        signal.append(channel_signal)

    signal = np.stack(signal, axis=1)
    return signal


def show_spectrum(win_name: str,
                  spectrums: np.ndarray,
                  samplerate: int,
                  call_plt_show: bool = True):
    """
    The show_spectrum function displays a spectrogram of the input signal.

    Note:
        The function takes three arguments:
        win_name - The name of the window that will be displayed.
        spectrums - A numpy array containing the spectrum data for each channel in an audio file
                    (or multiple channels). Each row represents one channel, and each column
                    represents a frequency bin across all channels. If there are more than
                    two dimensions, then it is assumed that there are multiple audio files being
                    passed in as an argument, and they will be plotted on separate subplots.
    Args:
        win_name: Set the title of the window
        spectrums: Pass the spectrums to be plotted
        samplerate: Audio samplerate
        call_plt_show: Enables/disables the interrupt, which displays a window with a graph
    """
    fig_spectogram = plt.figure(win_name, figsize=(15, 7.5))
    if len(spectrums.shape) > 2:
        for ch_idx, ch_spectrums in enumerate(spectrums):
            timebins, freqbins = np.shape(ch_spectrums)
            frequencies = np.abs(np.fft.fftfreq(freqbins*2, 1./samplerate)[:freqbins+1])
            ims = 20. * np.log10(np.abs(ch_spectrums) / 10e-6) # amplitude to decibel
            # Add subbplot data
            plt_spectogram = fig_spectogram.add_subplot(2, 1, ch_idx+1)
            plt_spectogram.set_title(f'Chanel[{ch_idx}]')
            im = plt_spectogram.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap='jet',
                                       interpolation="none")
            fig_spectogram.colorbar(im)
            plt_spectogram.set_xlabel("Time")
            plt_spectogram.set_ylabel("Frequency (Hz)")
            plt_spectogram.set_xlim([0, timebins-1])
            plt_spectogram.set_ylim([0, freqbins])
            # Calculate frequencies
            ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
            plt_spectogram.set_yticks(ylocs, ["%.02f" % frequencies[i] for i in ylocs])
    else:
        timebins, freqbins = np.shape(spectrums)
        ims = 20. * np.log10(np.abs(spectrums) / 10e-6) # amplitude to decibel
        plt.imshow(ims.T, origin="lower", aspect="auto", cmap='jet', interpolation="none")
        plt.colorbar()
        plt.xlabel("Time")
        plt.ylabel("Frequency (hz)")
        plt.xlim([0, timebins-1])
        plt.ylim([0, freqbins])

    if call_plt_show:
        plt.show()


def show_signal(win_name: str,
                signal: np.ndarray,
                call_plt_show: bool = True):
    """
    The show_signal function displays the signal in a new window.

    Note:
        The show_signal function accepts two arguments: win_name and signal.
        win_name is a string that will be used as the title of the plot window, 
        and signal is an array containing one or more signals to be plotted. It is
        important to call last show_signal() or show_spectrum() with call_plt_show: bool = True
        because otherwise they will not be displayed.

    Args:
        win_name: Set the title of the window
        signal: Pass a signal to be plotted
        call_plt_show: Enables/disables the interrupt, which displays a window with a graph
    """
    fig_signal = plt.figure(win_name, figsize=(15, 7.5))
    if len(signal.shape) > 1:
        for ch_idx, ch_signal in enumerate(signal.T):
            plt_signal = fig_signal.add_subplot(2, 1, ch_idx+1)
            plt_signal.set_title(f'Chanel[{ch_idx}]')
            plt_signal.plot(np.arange(ch_signal.shape[0]), ch_signal)
    else:
        plt.plot(np.arange(signal.shape[0]), signal)

    if call_plt_show:
        plt.show()


def set_samplerate(wav_file_path: Union[Path, str],
                   save_file_path: Union[Path, str],
                   samplerate: int):
    """
    The set_sample_rate function takes a wav file path and
    resamples it to the specified sample rate.
    The function returns an array of samples for the new audio file.
    
    Args:
        wav_file_path: Specify the path to the wav file that is going to be changed 
        save_file_path: Specify the path where you want to save your file
        samplerate: Set the sample rate of the audio file
    """
    sound = AudioSegment.from_file(wav_file_path, format='wav')
    sound = sound.set_frame_rate(samplerate)
    sound.export(save_file_path, format='wav')


def read_mp3_file(mp3_file_path: Union[Path, str],
                  samplerate: Optional[int] = None) -> Tuple[int, np.ndarray]:
    """
    The read_audio function reads a mp3 or wav file.

    Args:
        mp3_file_path: A source mp3 file
        samplerate: Sample rate conversion

    Returns: 
        Samplerate and an array of samples
    """
    audio_file_path = Path(mp3_file_path)
    wav_file_path = audio_file_path.parent / (audio_file_path.stem + '.wav')

    if not wav_file_path.exists():
        sound = AudioSegment.from_mp3(audio_file_path)
        sound.export(wav_file_path, format="wav")
    if samplerate is not None:
        converted_wav_file_path = audio_file_path.parent / \
                                  (audio_file_path.stem + f'_[{samplerate}].wav')
        if not converted_wav_file_path.exists():
            set_samplerate(wav_file_path, converted_wav_file_path, samplerate)
        return wavfile.read(converted_wav_file_path)
    return wavfile.read(wav_file_path)


def read_audio_file(audio_file_path: Union[Path, str],
                    samplerate: Optional[int] = None) -> Tuple[int, np.ndarray]:
    """
    Read the audio file. There are supported formats: wav, mp3.
    If the input format is not supported that the function will raise RuntimeError.

    Args:
        audio_file_path: A source audio file
        samplerate: Sample rate conversion
    
    Returns:
        Samplerate and an array of samples
    """
    
    audio_file_path = Path(audio_file_path)
    if audio_file_path.suffix == '.wav':
        if samplerate is not None:
            converted_wav_file_path = audio_file_path.parent / \
                                      (audio_file_path.stem + f'_[{samplerate}].wav')
            if not converted_wav_file_path.exists():
                set_samplerate(audio_file_path, converted_wav_file_path, samplerate)
            return wavfile.read(converted_wav_file_path)
        return wavfile.read(audio_file_path)
    elif audio_file_path.suffix == '.mp3':
        return read_mp3_file(audio_file_path, samplerate)
    raise RuntimeError(f'Unknown audio format of the file: {audio_file_path}')


def test_wav_file_2_spectogramm(audio_file_path: Union[Path, str],
                                frame_size: int = 2**10,
                                samplerate: Optional[int] = None,
                                cutoff: int = 0):
    """
    The test_wav_file_2_spectogramm function takes a wav file and converts it to the spectogramm
    and back.

    Note:
        The function also shows the input signal, output signal and spectrum of the input
        signal. In addition, we consider the difference between the signals to prove that
        the conversions work with minimal losses.

    Args:
        audio_file_path: Specify the path to the wav file
        frame_size: The size of the frame
        samplerate: Sample rate conversion
        cutoff: Number of frequencies to be trimmed/cutted
    """
    samplerate, in_signal = read_audio_file(audio_file_path, samplerate)

    t_start = time.time()
    spectrums, source_slice = stft(in_signal, frame_size)
    t_mid = time.time()
    out_signal = istft(spectrums, source_slice)
    t_end = time.time()

    signal_diff = np.abs(in_signal - out_signal)
    offset = 1e-6 - min(in_signal.min(), out_signal.min())
    diff_sum = np.mean(signal_diff / (np.maximum(in_signal, out_signal) + offset))

    print(f'Diff sum = {diff_sum} '
          f'Time=FFT {round(t_mid - t_start, 3)} + '
          f'IFFT {round(t_end - t_mid, 3)} = {round(t_end - t_start, 3)}[s]')

    wavfile.write(f"output.wav", samplerate, out_signal.astype(np.int16))

    # Important! call_plt_show: bool = True only in the last function
    show_signal("Signal input", in_signal, False)
    show_signal("Signal output", out_signal, False)
    show_spectrum(f"Spectrum", spectrums, samplerate, False)

    signal_diff = np.abs(in_signal - out_signal)
    show_signal("Signal difference", signal_diff, True)


if __name__ == "__main__":
    test_wav_file_2_spectogramm('test.wav', 2**10, None, 400)
