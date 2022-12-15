import numpy as np


def diff(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Take the finite difference of a tensor along an axis.
    
    Args:
        x: Input tensor of any dimension.
        axis: Axis on which to take the finite difference.
    Returns:
        Tensor with size less than x by 1 along the difference dimension.
    
    Raises:
        ValueError: Axis out of range for tensor.
    """

    shape = list(x.shape)

    begin_back = [0 for _ in range(len(shape))]
    begin_front = [0 for _ in range(len(shape))]

    begin_front[axis] = 1
    shape[axis] -= 1

    slice_front = x[begin_front[0]:begin_front[0]+shape[0], begin_front[1]:begin_front[1]+shape[1]]
    slice_back = x[begin_back[0]:begin_back[0]+shape[0], begin_back[1]:begin_back[1]+shape[1]]

    return slice_front - slice_back


def unwrap_phases(phases: np.ndarray,
                  discont: float = np.pi,
                  axis: int = 0) -> np.ndarray:
    """
    Unwrap a cyclical phase tensor.
    
    Args:
        phases: Phase tensor.
        discont: Size of the cyclic discontinuity.
        axis: Axis of which to unwrap.
        
    Returns:
        Unwrapped tensor of same size as input.
    """

    d_phases = diff(phases, axis=axis)
    d_phases_mod = np.mod(d_phases + np.pi, 2 * np.pi) - np.pi

    idx = np.logical_and(np.equal(d_phases_mod, -np.pi), np.greater(d_phases, 0))
    d_phases_mod = np.where(idx, np.ones_like(d_phases_mod) * np.pi, d_phases_mod)
    phase_correction = d_phases_mod - d_phases
    
    idx = np.less(np.abs(d_phases), discont)
    d_phases_mod = np.where(idx, np.zeros_like(d_phases), d_phases)  # ?
    phase_cumsum = np.cumsum(phase_correction, axis=axis)
    
    shape = list(phases.shape)
    shape[axis] = 1
    phase_cumsum = np.concatenate([np.zeros(shape, dtype=phases.dtype), phase_cumsum], axis=axis)
    return phases + phase_cumsum


def wrap_phases(unwrapped_phases: np.ndarray) -> np.ndarray:
    """
    Warp phases to range [-pi, pi]

    Args:
        unwrapped_phases

    Returns:
        Wrapped tensor of same size as input.
    """

    return np.mod(unwrapped_phases + np.pi, 2 * np.pi) - np.pi


def instantaneous_frequency(spectra: np.ndarray,
                            copy: bool = False) -> np.ndarray:
    """
    Transform a fft tensor from phase angle to instantaneous frequency.
    Unwrap and take the finite difference of the phase. Pad with initial phase to
    keep the tensor the same size.

    Args:
        spectra: Tensor of angles in radians. [Channels, Time, Freqs]
        copy: Do computation on the input array or not.

    Returns:
        Instantaneous frequency (derivative of phase). Same size as input.
    """

    time_axis = 1

    if copy:
        spectra = spectra.copy()

    unwrapped_phases = unwrap_phases(spectra[1::2, :, :], axis=time_axis)
    d_phase = diff(unwrapped_phases, axis=time_axis)
    shape = list(unwrapped_phases.shape)

    shape[time_axis] = 1
    begin = [0 for _ in range(len(shape))]
    phase_slice = unwrapped_phases[begin[0]:begin[0]+shape[0],
                                   begin[1]:begin[1]+shape[1],
                                   begin[2]:begin[2]+shape[2]]
    d_phase = np.concatenate([phase_slice, d_phase], axis=time_axis) / np.pi
    spectra[1::2, :, :] = d_phase
    return spectra


def inverse_instantaneous_frequency(spectra: np.ndarray,
                                    copy: bool = False) -> np.ndarray:
    """
    An inverse function for 'instantaneous_frequency'

    Args:
        spectra: Tensor of angles in radians. [Channels, Time, Freqs]
        copy: Do computation on the input array or not.

    Returns:
        The original spectra
    """

    time_axis = 1

    if copy:
        spectra = spectra.copy()
    spectra[1::2, :, :] = np.cumsum(spectra[1::2, :, :], axis=time_axis) * np.pi
    spectra[1::2, :, :] = wrap_phases(spectra[1::2, :, :])
    return spectra