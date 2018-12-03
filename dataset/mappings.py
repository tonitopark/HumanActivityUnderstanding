import numpy as np
import torch
from PIL import Image


class ComposeMappings(object):

    def __init__(self, mappings):
        self.mappings = mappings

    def __call__(self, image):
        for mapping in self.mappings:
            mapping.randomize()
            image = mapping(image)

        return image


class NormalizeFrame(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image_tensor):
        for image, mean, std in zip(image_tensor, self.mean, self.std):
            image.sub_(mean).div_(std)

        return image_tensor

    def randomize(self):
        pass

class NormalizeFrameToUnity(object):

    def __init__(self,lb,ub):
        self.lb=lb
        self.up=ub

    def __call__(self,image_tensor):
        for image in image_tensor:
            image.mul_(2*self.up).add_(1*self.lb)

        return image_tensor

    def randomize(self):
        pass




class FlipFrame(object):

    def __init__(self):
        self.flip_probability = np.random.rand(1)

    def __call__(self, image_pil):

        if self.flip_probability < 0.5:
            return image_pil.transpose(Image.FLIP_LEFT_RIGHT)

        else:
            return image_pil

    def randomize(self):
        self.flip_probability = np.random.rand(1)


class ResizeFrame(object):

    def __init__(self, size, intepolation_method=Image.BILINEAR):

        self.resize_size = size
        self.interpolation_method = intepolation_method

    def __call__(self, image_pil):

        width, height = image_pil.size

        # when given a single int value shorter edge will match the self.size
        if isinstance(self.resize_size, int):

            if (width <= height and width == self.resize_size) or \
                    (height <= width and height == self.resize_size):
                return image_pil

            if width <= height:
                return image_pil.resize((self.resize_size,
                                         int(self.resize_size * height / width)),
                                        self.interpolation_method)

            else:
                return image_pil.resize((int(self.resize_size * width / height),
                                         self.resize_size),
                                        self.interpolation_method)
        # given a pair
        else:
            return image_pil.resize(self.resize_size, self.interpolation_method)

    def randomize(self):
        pass


class CropFramePart(object):

    def __init__(self, size, crop_method=None):

        if isinstance(size, int):
            self.crop_size = (int(size), int(size))
        else:
            self.crop_size = size

        self.crop_method = crop_method

        self.crop_method_names = ['center', 'topleft', 'topright', 'bottomleft', 'bottomright']

        if crop_method is None:
            self.select_position_at_random = True
            self.randomize()
        else:
            self.select_position_at_random = False

    def __call__(self, image_pil):

        # import matplotlib.pyplot as plt
        # plt.imshow(image_pil)
        # plt.show()

        width, height = image_pil.size
        x_topleft = 0
        y_topleft = 0

        if self.crop_method == 'center':
            x_topleft = int(round(width/2. - self.crop_size[0] / 2.))
            y_topleft = int(round(height/2. - self.crop_size[1] / 2.))

        elif self.crop_method == 'topleft':
            x_topleft = 0
            y_topleft = 0

        elif self.crop_method == 'topright':
            x_topleft = width - self.crop_size[0]
            y_topleft = 0

        elif self.crop_method == 'bottomleft':
            x_topleft = 0
            y_topleft = height - self.crop_size[1]

        elif self.crop_method == 'bottonright':
            x_topleft = width - self.crop_size[0]
            y_topleft = width - self.crop_size[1]

        return image_pil.crop((x_topleft,
                               y_topleft,
                               x_topleft + self.crop_size[0],
                               y_topleft + self.crop_size[1]))



    def randomize(self):
        if self.select_position_at_random:
            idx = np.random.randint(0, len(self.crop_method_names) - 1)
            self.crop_method = self.crop_method_names[idx]


class ToTensor(object):

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, image_pil):

        if isinstance(image_pil, np.ndarray):
            # handle numpy array
            image_tensor = torch.from_numpy(image_pil.transpose((2, 0, 1)))
            # backward compatibility
            return image_tensor.float().div(self.norm_value)

        # handle PIL Image
        if image_pil.mode is not None:
            if image_pil.mode == 'I':
                image_tensor = torch.from_numpy(np.array(image_pil, np.int32, copy=False))
            elif image_pil.mode == 'I;16':
                image_tensor = torch.from_numpy(np.array(image_pil, np.int16, copy=False))
            else:
                image_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image_pil.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if image_pil.mode == 'YCbCr':
                nchannel = 3
            elif image_pil.mode == 'I;16':
                nchannel = 1
            else:
                nchannel = len(image_pil.mode)
            image_tensor = image_tensor.view(image_pil.size[1], image_pil.size[0], nchannel)

            # TO [Channel,Height,Width] format
            image_tensor = image_tensor.transpose(0, 1).transpose(0, 2).contiguous()
            if isinstance(image_tensor, torch.ByteTensor):
                return image_tensor.float().div(self.norm_value)
            else:
                return image_tensor

    def randomize(self):
        pass
