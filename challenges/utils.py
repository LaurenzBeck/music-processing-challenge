def get_timestamps_for_framerate(fps, length):
    return [frame_index / fps + 1 / (2 * fps) for frame_index in range(length)]
