from __future__ import division
import random, math, sys, collections, warnings
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision

if sys.version_info < (3, 3):
	Iterable = collections.Iterable
else:
	Iterable = collections.abc.Iterable

_pil_interpolation_to_str = {
	Image.NEAREST: 'PIL.Image.NEAREST',
	Image.BILINEAR: 'PIL.Image.BILINEAR',
	Image.BICUBIC: 'PIL.Image.BICUBIC',
	Image.LANCZOS: 'PIL.Image.LANCZOS',
	Image.HAMMING: 'PIL.Image.HAMMING',
	Image.BOX: 'PIL.Image.BOX',
}

_pil_interpolation_to_mode = {
	Image.NEAREST: 'nearest',
	Image.BILINEAR: 'bilinear',
	Image.BICUBIC: 'bicubic'
}

Compose = torchvision.transforms.Compose

class Normalize(object):
	def __init__(self, mean, std, inplace=False):
		self.mean = mean
		self.std = std
		self.inplace = inplace

	def __call__(self, tensor):
		return normalize(tensor, self.mean, self.std, self.inplace)

	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def normalize(tensor, mean, std, inplace=False):
	if not torch.is_tensor(tensor):
		raise TypeError('tensor should be Tensor. Got {}'.format(type(tensor)))

	if not inplace:
		tensor = tensor.clone()

	dtype = tensor.dtype
	mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
	std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
	tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
	return tensor

class RandomHorizontalFlip(object):
	def __init__(self, p=0.5):
		self.p = p

	def __call__(self, img):
		if random.random() < self.p:
			return hflip(img)
		return img

	def __repr__(self):
		return self.__class__.__name__ + '(p={})'.format(self.p)

def hflip(img):
	if not torch.is_tensor(img):
		raise TypeError('img should be Tensor. Got {}'.format(type(img)))

	return img.flip([3])

class RandomResizedCrop(object):
	def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
		if isinstance(size, tuple):
			self.size = size
		else:
			self.size = (size, size)
		if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
			warnings.warn("range should be of kind (min, max)")

		self.interpolation = interpolation
		self.scale = scale
		self.ratio = ratio

	@staticmethod
	def get_params(img, scale, ratio):
		area = img.size(2) * img.size(3)

		for attempt in range(10):
			target_area = random.uniform(*scale) * area
			log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
			aspect_ratio = math.exp(random.uniform(*log_ratio))

			w = int(round(math.sqrt(target_area * aspect_ratio)))
			h = int(round(math.sqrt(target_area / aspect_ratio)))

			if w <= img.size(3) and h <= img.size(2):
				i = random.randint(0, img.size(2) - h)
				j = random.randint(0, img.size(3) - w)
				return i, j, h, w

		in_ratio = img.size(3) / img.size(2)
		if (in_ratio < min(ratio)):
			w = img.size(3)
			h = int(round(w / min(ratio)))
		elif (in_ratio > max(ratio)):
			h = img.size(2)
			w = int(round(h * max(ratio)))
		else:
			w = img.size(3)
			h = img.size(2)
		i = (img.size(2) - h) // 2
		j = (img.size(3) - w) // 2
		return i, j, h, w

	def __call__(self, img):
		i, j, h, w = self.get_params(img, self.scale, self.ratio)
		return resized_crop(img, i, j, h, w, self.size, self.interpolation)

	def __repr__(self):
		interpolate_str = _pil_interpolation_to_str[self.interpolation]
		format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
		format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
		format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
		format_string += ', interpolation={0})'.format(interpolate_str)
		return format_string

def resized_crop(img, i, j, h, w, size, interpolation=Image.BILINEAR):
	if not torch.is_tensor(img):
		raise TypeError('img should be Tensor. Got {}'.format(type(img)))

	img = crop(img, i, j, h, w)
	img = resize(img, size, interpolation)
	return img

def crop(img, i, j, h, w):
	if not torch.is_tensor(img):
		raise TypeError('img should be Tensor. Got {}'.format(type(img)))

	return img[:, :, i:(i+h), j:(j+w)]

class Resize(object):
	def __init__(self, size, interpolation=Image.BILINEAR):
		assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
		self.size = size
		self.interpolation = interpolation

	def __call__(self, img):
		return resize(img, self.size, self.interpolation)

	def __repr__(self):
		interpolate_str = _pil_interpolation_to_str[self.interpolation]
		return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

def resize(img, size, interpolation=Image.BILINEAR):
	if not torch.is_tensor(img):
		raise TypeError('img should be Tensor. Got {}'.format(type(img)))
	if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
		raise TypeError('Got inappropriate size arg: {}'.format(size))

	if isinstance(size, int):
		_, _, h, w = img.size()
		if (w <= h and w == size) or (h <= w and h == size):
			return img
		if w < h:
			ow = size
			oh = int(size * h / w)
			return F.interpolate(img, size=(oh, ow), mode=_pil_interpolation_to_mode[interpolation], align_corners=False)
		else:
			oh = size
			ow = int(size * w / h)
			return F.interpolate(img, size=(oh, ow), mode=_pil_interpolation_to_mode[interpolation], align_corners=False)
	else:
		return F.interpolate(img, size=size, mode=_pil_interpolation_to_mode[interpolation], align_corners=False)

class ToTensor(object):
	def __call__(self, pic):
		return to_tensor(pic)

	def __repr__(self):
		return self.__class__.__name__ + '()'

def to_tensor(pic):
	if not isinstance(pic[0], Image.Image):
		raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

	if pic[0].mode != 'RGB':
		raise TypeError('pic.mode should be RGB. Got {}'.format(pic.mode))

	img = torch.stack([torch.ByteTensor(torch.ByteStorage.from_buffer(p.tobytes())) for p in pic])
	img = img.view(len(pic), pic[0].size[1], pic[0].size[0], 3)
	img = img.transpose(1, 2).transpose(1, 3).contiguous()

	if isinstance(img, torch.ByteTensor):
		return img.float().div(255)
	else:
		return img
