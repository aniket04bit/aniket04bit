import cv2
import numpy as np
import torch
from PIL import Image
import math


class RecResizeImg(object):
    def __init__(self, image_shape, infer_mode=False, character_dict_path=None, padding=True, **kwargs):
        self.image_shape = image_shape
        self.infer_mode = infer_mode
        self.character_dict_path = character_dict_path
        self.padding = padding

    def __call__(self, data):
        img = data['image']
        if self.infer_mode and self.character_dict_path is not None:
            norm_img, valid_ratio = resize_norm_img(img, self.image_shape, self.padding)
            data['image'] = norm_img
            data['valid_ratio'] = valid_ratio
            return data
        norm_img = resize_norm_img_chinese(img, self.image_shape)
        data['image'] = norm_img
        return data


def resize_norm_img(img, image_shape, padding=True):
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    if not padding:
        resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_w = imgW
    else:
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio


def resize_norm_img_chinese(img, image_shape):
    imgC, imgH, imgW = image_shape
    # todo: change to 0 and modified image shape
    max_wh_ratio = imgW * 1.0 / imgH
    h, w = img.shape[:2]
    ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, ratio)
    imgW = int(imgH * max_wh_ratio)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        assert order in ['hwc', 'chw']
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        self.order = order
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"
        
        data['image'] = (img.astype('float32') * self.scale - self.mean) / self.std
        return data


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        data['image'] = img.transpose((2, 0, 1))
        return data


class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


class RecAug(object):
    def __init__(self, tia_prob=0.4, crop_prob=0.4, **kwargs):
        self.tia_prob = tia_prob
        self.crop_prob = crop_prob

    def __call__(self, data):
        img = data['image']
        h, w, _ = img.shape

        # TIA (Text Image Augmentation)
        if np.random.rand() < self.tia_prob:
            img = self.tia_distort(img, np.random.randint(3))

        # Random Crop
        if np.random.rand() < self.crop_prob:
            img = self.random_crop(img)

        data['image'] = img
        return data

    def tia_distort(self, img, distort_type):
        # Simple TIA implementation
        h, w, _ = img.shape
        
        if distort_type == 0:  # Horizontal distortion
            map_x = np.zeros((h, w), dtype=np.float32)
            map_y = np.zeros((h, w), dtype=np.float32)
            for y in range(h):
                for x in range(w):
                    map_x[y, x] = x + np.sin(2 * np.pi * y / h) * 3
                    map_y[y, x] = y
            img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        elif distort_type == 1:  # Vertical distortion
            map_x = np.zeros((h, w), dtype=np.float32)
            map_y = np.zeros((h, w), dtype=np.float32)
            for y in range(h):
                for x in range(w):
                    map_x[y, x] = x
                    map_y[y, x] = y + np.sin(2 * np.pi * x / w) * 2
            img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        # Type 2: no distortion
        
        return img

    def random_crop(self, img):
        h, w, _ = img.shape
        crop_h = int(h * (0.8 + np.random.rand() * 0.2))
        crop_w = int(w * (0.8 + np.random.rand() * 0.2))
        
        start_y = np.random.randint(0, h - crop_h + 1)
        start_x = np.random.randint(0, w - crop_w + 1)
        
        cropped = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
        return cv2.resize(cropped, (w, h))


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator, dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops


def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_train_transforms():
    """Create training transforms for PP-OCRv3."""
    transforms = [
        {'RecAug': {}},
        {'RecResizeImg': {'image_shape': [3, 48, 320]}},
        {'KeepKeys': {'keep_keys': ['image', 'label']}}
    ]
    return create_operators(transforms)


def create_eval_transforms():
    """Create evaluation transforms for PP-OCRv3."""
    transforms = [
        {'RecResizeImg': {'image_shape': [3, 48, 320], 'padding': True}},
        {'KeepKeys': {'keep_keys': ['image', 'label']}}
    ]
    return create_operators(transforms)