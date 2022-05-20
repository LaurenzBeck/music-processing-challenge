def get_timestamps_for_framerate(fps, length):
    return [frame_index / fps for frame_index in range(length)]
