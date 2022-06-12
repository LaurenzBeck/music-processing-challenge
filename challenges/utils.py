def get_timestamps_for_framerate(fps, length):
    return [frame_index / fps + 1 / (2 * fps) for frame_index in range(length)]


def get_index_from_timestamp(timestamp, sampling_rate=44100):
    return int(timestamp * sampling_rate)


def get_timestamp_from_index(idx, sampling_rate=44100):
    return idx / sampling_rate
