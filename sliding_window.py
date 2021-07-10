import numpy as np


def sliding_window(data, length_in_seconds=1, sampling_rate=50, overlap_ratio=None):
    """
    # overlap_ratio:
    # index results via windows[window_no][entry_no] (same for indices)
    :param data: input array, can be numpy or pandas dataframe
    :param length_in_seconds: window length as seconds
    :param sampling_rate: sampling rate in hertz as integer value
    :param overlap_ratio: overlap is meant as percentage and should be an integer value
    :return: tuple of windows and indices
    """
    # try:
    #     data = np.array(data)
    # except:
    #     pass
    windows = []
    indices = []
    curr = 0
    overlapping_elements = 0
    win_len = int(length_in_seconds * sampling_rate)
    if overlap_ratio != None:
        overlapping_elements = int((overlap_ratio / 100) * (win_len))
        if overlapping_elements >= win_len:
            print('Number of overlapping elements exceeds window size.')
            return
    while (curr < len(data) - win_len):
        windows.append(data[curr:curr + win_len])
        indices.append([curr, curr + win_len])
        curr = curr + win_len - overlapping_elements

    return np.array(windows), np.array(indices)


def sliding_window_samples(data, samples_per_window, overlap_ratio):
    """
    # overlap_ratio:
    # index results via windows[window_no][entry_no] (same for indices)
    :param data: input array, can be numpy or pandas dataframe
    :param samples_per_window: number of samples in each window
    :param overlap_ratio: overlap is meant as percentage and should be an integer value
    :return: tuple of windows and indices
    """

    windows = []
    indices = []
    curr = 0

    win_len = samples_per_window
    if overlap_ratio != None:
        overlapping_elements = int((overlap_ratio / 100) * (win_len))
        if overlapping_elements >= win_len:
            print('Number of overlapping elements exceeds window size.')
            return
    while (curr < len(data) - win_len):
        windows.append(data[curr:curr + win_len])
        indices.append([curr, curr + win_len])
        curr = curr + win_len - overlapping_elements

    try:
        result_windows = np.array(windows)
        result_indices = np.array(indices)
    except:
        result_windows = np.empty(shape=(len(windows), win_len, data.shape[1]), dtype=object)
        result_indices = np.array(indices)
        for i in range(0, len(windows)):
            result_windows[i] = windows[i]
            result_indices[i] = indices[i]

    # result_windows = np.array(windows)
    # result_indices = np.array(indices)

    return result_windows, result_indices
