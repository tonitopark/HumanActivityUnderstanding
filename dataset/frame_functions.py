import numpy as np


class PadFrames(object):

    def __init__(self, length):
        self.clip_length = length

    def __call__(self, frame_indices):

        indices = frame_indices

        for index in indices:
            if len(indices) >= self.clip_length:
                break
            indices.append(index)

        return indices


class SelectFrames(object):

    def __init__(self, length):
        self.clip_length = length

    def __call__(self, frame_indices, method):

        if method == 'begin':

            begin_index = 0
            end_index = self.clip_length

        elif method == 'center':

            center_index = len(frame_indices) // 2
            begin_index = max(0, center_index - (self.clip_length // 2))
            end_index = min(begin_index + self.clip_length, len(frame_indices))

        elif method == 'random':

            candidate_max_value = max(1, len(frame_indices) - self.clip_length - 1)
            begin_index = np.random.randint(low = 0, high = candidate_max_value)
            end_index = min(len(frame_indices), begin_index + self.clip_length)
            #print('max_{}_,min_{}__beging_{}_end_{}'.format(candidate_max_value, np.max(frame_indices),begin_index,end_index))


        indices = frame_indices[begin_index:end_index]

        for index in indices:
            if len(indices) >= self.clip_length:
                break
            indices.append(index)

        return indices
