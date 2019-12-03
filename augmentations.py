import torch
import torch.nn.functional as F


class RandomPaddedCrop:
    def __init__(self, pad_size):
        self.pad_size = pad_size

    def __call__(self, x):
        img_h, img_w = x.shape[-2:]
        x = F.pad(x, [self.pad_size] * 4, mode='reflect')
        crop_height = torch.randint(0, self.pad_size * 2, (1,)).item()
        crop_width = torch.randint(0, self.pad_size * 2, (1,)).item()
        return x[:, :, crop_height:crop_height + img_h, crop_width:crop_width + img_w]


class RandomHorizontalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        if torch.rand((1,)).item() < self.p:
            return x.flip(-1)
        return x


class GaussianNoise:
    def __init__(self, noise_rate):
        self.noise_rate = noise_rate

    def __call__(self, x):
        x += torch.randn_like(x) * self.noise_rate
        return x


class Augmentor:
    def __init__(self, config):
        self.config = config
        if config.augmentations.use:
            if config.augmentations.use_crop:
                self.crop = RandomPaddedCrop(config.augmentations.pad_size)
            if config.augmentations.use_flip:
                self.flip = RandomHorizontalFlip(config.augmentations.flip_prob)
            if config.augmentations.use_noise:
                self.noise = GaussianNoise(config.augmentations.noise_rate)

    def __call__(self, x):
        if not self.config.augmentations.use:
            return x
        out = []
        for i in range(len(x)):
            cur_img = x[i:i+1]
            if self.config.augmentations.use_crop:
                cur_img = self.crop(cur_img)
            if self.config.augmentations.use_flip:
                cur_img = self.flip(cur_img)
            if self.config.augmentations.use_noise:
                cur_img = self.noise(cur_img)
                out.append(cur_img)
        return torch.cat(out)
