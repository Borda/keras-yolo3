"""Miscellaneous utility functions."""

import os
import logging
import warnings
import gc
from functools import reduce, partial, wraps
import multiprocessing as mproc

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from pathos.multiprocessing import ProcessPool

CPU_COUNT = mproc.cpu_count()


def nb_workers(ratio):
    if not ratio:
        return 1
    nb = ratio if isinstance(ratio, int) else int(CPU_COUNT * ratio)
    return max(1, nb)


def update_path(my_path, max_depth=5, abs_path=True):
    """ update path as bobble up strategy

    :param str my_path:
    :param int max_depth:
    :param bool abs_path:
    :return:

    >>> os.path.isdir(update_path('model_data'))
    True
    """
    if not my_path or my_path.startswith('/'):
        return my_path
    elif my_path.startswith('~'):
        return os.path.expanduser(my_path)

    up_path = my_path
    for _ in range(max_depth):
        if os.path.exists(up_path):
            my_path = up_path
            break
        up_path = os.path.join('..', up_path)

    if abs_path:
        my_path = os.path.abspath(my_path)
    return my_path


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    """resize image with unchanged aspect ratio using padding

    >>> img = Image.fromarray(np.random.randint(0, 255, (800, 600, 3)).astype(np.uint8))
    >>> letterbox_image(img, (416, 416)).size
    (416, 416)
    """
    iw, ih = image.size
    w, h = size
    scale = min(float(w) / iw, float(h) / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def _rand(a=0, b=1):
    """ random number in given range

    :param float a: any number
    :param float b: any number
    :return float:

    >>> 0 <= _rand() <= 1
    True
    >>> np.random.seed(0)
    >>> _rand(1, 1)
    1
    """
    low = min(a, b)
    high = max(a, b)
    if low == high:
        return low
    return np.random.rand() * (high - low) + low


def io_image_decorate(func):
    """ costume decorator to suppers debug messages from the PIL function
    to suppress PIl debug logging
    - DEBUG:PIL.PngImagePlugin:STREAM b'IHDR' 16 13
    :param func:
    :return:
    """

    @wraps(func)
    def wrap(*args, **kwargs):
        log_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.INFO)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            response = func(*args, **kwargs)
        logging.getLogger().setLevel(log_level)
        return response

    return wrap


@io_image_decorate
def image_open(path_img):
    """ just a wrapper to suppers debug messages from the PIL function
    to suppress PIl debug logging - DEBUG:PIL.PngImagePlugin:STREAM b'IHDR' 16 13
    :param str path_img:
    :return Image:

    >>> path_img = os.path.join(update_path('model_data'), 'bike-car-dog.jpg')
    >>> image_open(path_img).size
    (520, 518)
    """
    return Image.open(update_path(path_img))


def augment_image_color(image, hue, sat, val):
    """Randomize image colour in HSV spectrum in given range.

    :param image: Input image
    :param float hue: range in +/-
    :param float sat: greater then 1
    :param float val: greater then 1
    :return:

    >>> img = image_open(os.path.join(update_path('model_data'), 'bike-car-dog.jpg'))
    >>> augment_image_color(img, 0.1, 1.1, 1.2).shape
    (518, 520, 3)
    """
    hue = _rand(-hue, hue)
    sat = _rand(1 - abs(1 - sat), sat)
    val = _rand(1 - abs(1 - val), val)

    img = rgb_to_hsv(np.array(image) / 255.)
    img[..., 0] += hue
    img[..., 0][img[..., 0] > 1] -= 1
    img[..., 0][img[..., 0] < 0] += 1
    img[..., 1] *= sat
    img[..., 2] *= val
    image_data = hsv_to_rgb(img)

    # numpy array, 0 to 1
    assert np.max(image_data) < 2.  # check the range is about (0, 1)
    image_data = np.clip(image_data, 0, 1)
    return image_data


def adjust_bboxes(bbox, input_shape, flip_horizontal, flip_vertical, scale_x, scale_y, dx, dy,
                  bbox_overlap=0.9):
    """randomize bounding boxes

    :param ndarray boxes:
    :param int max_boxes:
    :param bool flip_horizontal:
    :param bool flip_vertical:
    :param int img_w: image width
    :param int img_h: image height
    :param int cnn_h: input CNN height
    :param int cnn_w: input CNN width
    :param int new_w:
    :param int new_h:
    :param int dx: translation in X axis
    :param int dy: translation in Y axis
    :param float bbox_overlap: threshold drop all boxes with lower overlap
    :return ndarray:

    >>> np.random.seed(0)
    >>> bboxes = np.array([[10, 15, 20, 25, 0], [3, 5, 4, 10, 1]])
    >>> adjust_bboxes(bboxes, (20, 20), True, False, 1., 1., 0, 0, bbox_overlap=0.2)
    array([[ 0, 15, 10, 20,  0],
           [16,  5, 17, 10,  1]])
    >>> adjust_bboxes(bboxes, (20, 20), False, True, 1., 1., 0, 0, bbox_overlap=0.2)
    array([[10,  0, 20,  5,  0],
           [ 3, 10,  4, 15,  1]])
    >>> adjust_bboxes(bboxes, (20, 20), False, False, 1.2, 0.8, 0, 0, bbox_overlap=0.2)
    array([[12, 12, 20, 20,  0],
           [ 3,  4,  4,  8,  1]])
    >>> adjust_bboxes(bboxes, (20, 20), False, False, 1., 1., -2, 3, bbox_overlap=0.2)
    array([[ 1,  8,  2, 13,  1]])
    """
    boxes = bbox.copy()  # make a copy
    bb_sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * scale_x * scale_y
    cnn_h, cnn_w = input_shape

    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x + dx
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y + dy
    if flip_horizontal:
        boxes[:, [0, 2]] = cnn_w - boxes[:, [2, 0]]
    if flip_vertical:
        boxes[:, [1, 3]] = cnn_h - boxes[:, [3, 1]]

    boxes = _filter_empty_bboxes(boxes, cnn_w, cnn_h, bb_sizes, bbox_overlap)
    return boxes


def normalize_image_bboxes(image, boxes, input_shape, resize_img,
                           allow_rnd_shift=True, bbox_overlap=0.9, interp=Image.BICUBIC):
    """normalize image bounding bbox

    :param Image image:
    :param ndarray boxes: bounding boxes
    :param tuple(int,int) input_shape:
    :param int max_boxes: maximum nb bounding boxes
    :param bool resize_img: allow resize image to CNN input shape
    :param bool allow_rnd_shift: allow shifting image not only centered crop
    :param int interp: image interpolation
    :param float bbox_overlap: threshold drop all boxes with lower overlap
    :return:

    >>> np.random.seed(0)
    >>> img = np.zeros((15, 20), dtype=np.uint8)
    >>> img[5:10, 10:15] = 255
    >>> img = Image.fromarray(img)
    >>> bboxes = np.array([[10, 15, 20, 25, 0], [3, 5, 4, 10, 1]])
    >>> image_data, box_data = normalize_image_bboxes(img, bboxes, (10, 10), resize_img=True,
    ...                                               interp=Image.NEAREST, bbox_overlap=0.1)
    >>> np.round(image_data, 1)[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 1. , 1. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 1. , 1. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 1. , 1. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    >>> box_data
    array([[ 5,  8, 10, 10,  0],
           [ 1,  3,  2,  6,  1]])
    >>> image_data, box_data = normalize_image_bboxes(img, bboxes, (25, 25), resize_img=True,
    ...                                             interp=Image.NEAREST)
    >>> np.round(image_data, 1)[:, :, 0]  # doctest: +ELLIPSIS
    array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , ..., 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           ...
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , ..., 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , ..., 1. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , ..., 1. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , ..., 1. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , ..., 1. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , ..., 1. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , ..., 1. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , ..., 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           ...
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , ..., 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    >>> box_data
    array([[ 3, 11,  5, 17,  1]])
    >>> image_data, box_data = normalize_image_bboxes(img, bboxes, (15, 15), resize_img=False,
    ...                                             interp=Image.NEAREST)
    >>> np.round(image_data, 1)[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    >>> box_data
    array([[ 2,  5,  3, 10,  1]])
    """
    if resize_img:
        img_data, scale, (dx, dy) = _scale_image_to_cnn(
            image, input_shape, allow_rnd_shift=allow_rnd_shift, interp=interp)
    else:
        img_data, scale, (dx, dy) = _crop_image_to_cnn(image, input_shape, allow_rnd_shift)

    if len(boxes) == 0:
        return img_data, np.zeros((1, 5))

    boxes = boxes.copy()
    np.random.shuffle(boxes)
    bb_sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * scale * scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + dx
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + dy

    boxes = _filter_empty_bboxes(boxes, *input_shape, bb_sizes=bb_sizes, bb_overlap=bbox_overlap)

    return img_data, boxes


def _scale_image_to_cnn(image, input_shape, scaling=1., allow_rnd_shift=True,
                        interp=Image.BICUBIC):
    """scale image to fit CNN input

    :param ndarray image: original image
    :param tuple(int,int) input_shape:CNN input size
    :param float scaling: scaling factor
    :param bool allow_rnd_shift: allow shifting image not only centered crop
    :param interp: image interpolation
    :return:
    """
    img_w, img_h = image.size
    cnn_h, cnn_w = input_shape

    scale = min(float(cnn_w) / img_w, float(cnn_h) / img_h) * scaling
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    dx, dy = _image_shift(cnn_w, cnn_h, new_w, new_h, allow_rnd_shift)

    image = image.resize((new_w, new_h), interp)

    new_image = Image.new('RGB', (cnn_w, cnn_h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    img_data = np.array(new_image) / 255.

    return img_data, scale, (dx, dy)


def _crop_image_to_cnn(image, input_shape, allow_rnd_shift=True):
    """crop image according the CNN input size

    :param ndarray image: original image
    :param tuple(int,int) input_shape:CNN input size
    :param allow_rnd_shift:
    :return:
    """
    img_w, img_h = image.size
    cnn_h, cnn_w = input_shape
    dx, dy = _image_shift(cnn_w, cnn_h, img_w, img_h, allow_rnd_shift)

    new_image = Image.new('RGB', (cnn_w, cnn_h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    img_data = np.array(new_image) / 255.

    return img_data, 1., (dx, dy)


def _image_shift(cnn_w, cnn_h, img_w, img_h, allow_rnd_shift):
    """compute position/shift for inserting image to CNN shape

    :param int cnn_w: CNN width
    :param int cnn_h: CNN height
    :param int img_w: image width
    :param int img_h: image height
    :param bool allow_rnd_shift:
    :return tuple(int,int): shift
    """
    diff_w = cnn_w - img_w
    diff_h = cnn_h - img_h
    if allow_rnd_shift:
        diff_w = diff_w // 2 if img_w > cnn_w else diff_w
        diff_h = diff_h // 2 if img_h > cnn_h else diff_h
        dx = int(_rand(0, diff_w))
        dy = int(_rand(0, diff_h))
    else:
        dx = diff_w // 2
        dy = diff_h // 2
    return dx, dy


def _copy_bboxes(boxes, adj_box_data, max_boxes, check_dropped=True):
    """copy the bounding boxes to preferred sized array

    :param list(list) boxes: input boxes
    :param ndarray adj_box_data: augmented boxes
    :param int|None max_boxes: maximal number of boxes
    :param bool check_dropped: show warning if the nb augmented boxes is lower then input
    :return:
    """
    box_data = np.zeros((max_boxes, 5))
    if check_dropped and len(adj_box_data) < len(boxes):
        logging.debug('Warning: %i of %i (%i%%) generated boxes was filtered out',
                      len(boxes) - len(adj_box_data), len(boxes),
                      int(float(len(boxes) - len(adj_box_data)) / len(boxes) * 100))
    nb_boxes = min(max_boxes, len(adj_box_data))
    box_data[:nb_boxes] = adj_box_data[:nb_boxes]
    return box_data


def _filter_empty_bboxes(boxes, cnn_w, cnn_h, bb_sizes, bb_overlap):
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, cnn_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, cnn_h)

    sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    mask = sizes > (bb_sizes * bb_overlap)
    # discard invalid box
    boxes = boxes[mask]
    return boxes


def get_augmented_data(annotation_line, input_shape, augment=True, max_boxes=20,
                       hue=.1, sat=1.5, val=1.5, jitter=0.3, img_scaling=1.2,
                       flip_horizontal=True, flip_vertical=False, resize_img=True,
                       allow_rnd_shift=True, bbox_overlap=0.95, interp=Image.BICUBIC):
    """augment pre-processing for real-time data augmentation

    :param str annotation_line:
    :param tuple(int,int) input_shape: CNN input size
    :param bool augment: perform augmentation
    :param int max_boxes: maximal number of training bounding boxes
    :param float jitter:
    :param float hue: range of change of HSV color HUE
    :param float sat: range of change of HSV color SAT
    :param float val: range of change of HSV color value
    :param float img_scaling: upper image scaling
    :param bool flip_horizontal: allow random flop image/boxes vertical
    :param bool flip_vertical: allow random flop image/boxes horizontal
    :param bool resize_img: resize image to fit fully to CNN
    :param bool allow_rnd_shift: allow shifting image not only centered crop
    :param float bbox_overlap: threshold in case cut image, drop all boxes with lower overlap
    :param int interp: image interpolation
    :return:

    >>> np.random.seed(0)
    >>> path_img = os.path.join(update_path('model_data'), 'bike-car-dog.jpg')
    >>> line = path_img + ' 100,150,200,250,0 300,50,400,200,1'
    >>> image_data, box_data = get_augmented_data(line, (416, 416), augment=True)
    >>> image_data.shape
    (416, 416, 3)
    >>> box_data  # doctest: +ELLIPSIS
    array([[243.,  39., 325., 162.,   1.],
           [ 80., 121., 162., 202.,   0.],
           [  0.,   0.,   0.,   0.,   0.],
           ...
           [  0.,   0.,   0.,   0.,   0.]])
    >>> image_data, box_data = get_augmented_data(line, (416, 416), augment=False)
    >>> image_data.shape
    (416, 416, 3)
    >>> box_data  # doctest: +ELLIPSIS
    array([[ 80., 120., 160., 200.,   0.],
           [240.,  40., 320., 160.,   1.],
           [  0.,   0.,   0.,   0.,   0.],
           ...
           [  0.,   0.,   0.,   0.,   0.]])
    """
    line_split = annotation_line.split()
    image = image_open(line_split[0])
    boxes = np.array([list(map(float, box.split(',')))
                      for box in line_split[1:]]).astype(int)

    # resize image
    # new_ar = cnn_w / cnn_h * _rand(1 - jitter, 1 + jitter) / _rand(1 - jitter, 1 + jitter)
    if resize_img:
        scaling = _rand(1 - abs(1 - img_scaling), img_scaling) if augment else 1.
        img_data, scaling, (dx, dy) = _scale_image_to_cnn(
            image, input_shape, scaling, allow_rnd_shift=allow_rnd_shift, interp=interp)
    else:
        img_data, scaling, (dx, dy) = _crop_image_to_cnn(image, input_shape, allow_rnd_shift)
    image = Image.fromarray(np.round(img_data * 255).astype(np.uint8))

    if augment:
        # flip image or not
        flip_horizontal = np.random.random() < 0.5 and flip_horizontal
        if flip_horizontal:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        flip_vertical = np.random.random() < 0.5 and flip_vertical
        if flip_vertical:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

        # distort image
        img_data = augment_image_color(image, hue, sat, val)
    else:
        flip_horizontal = False
        flip_vertical = False

    if len(boxes) == 0:
        max_boxes = max_boxes if max_boxes else 1
        return img_data, np.zeros((max_boxes, 5))

    np.random.shuffle(boxes)
    # NOTE: due to some randomisation some boxed can be out and filtered out
    adj_box_data = adjust_bboxes(boxes, input_shape, flip_horizontal, flip_vertical,
                                 scaling, scaling, dx, dy, bbox_overlap)
    box_data = _copy_bboxes(boxes, adj_box_data, max_boxes)
    # yolo3.visual.show_augment_data(image_open(line_split[0]), boxes, img_data, box_data, title=line_split[0])
    return img_data, box_data


def get_class_names(path_classes):
    logging.debug('loading classes from "%s"', path_classes)
    with open(path_classes) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_dataset_class_names(path_train_annot, path_classes=None):
    logging.debug('loading training dataset from "%s"', path_train_annot)
    with open(path_train_annot) as fp:
        lines = fp.readlines()
    classes = []
    for ln in lines:
        classes += [bbox.split(',')[-1] for bbox in ln.rstrip().split(' ')[1:]]
    uq_classes = sorted(set([int(c) for c in classes]))
    if path_classes and os.path.isfile(path_classes):
        cls_names = get_class_names(path_classes)
        uq_classes = {cls: cls_names[cls] for cls in uq_classes}
    return uq_classes


def get_nb_classes(path_train_annot=None, path_classes=None):
    if path_classes is not None and os.path.isfile(path_classes):
        class_names = get_class_names(path_classes)
        nb_classes = len(class_names)
    elif path_train_annot is not None and os.path.isfile(path_train_annot):
        uq_classes = get_dataset_class_names(path_train_annot)
        nb_classes = len(uq_classes)
    else:
        logging.warning('No input for extracting classes.')
        nb_classes = 0
    return nb_classes


def get_anchors(path_anchors):
    """loads the anchors from a file

    :param str path_anchors:

    >>> path_csv = os.path.join(update_path('model_data'), 'yolo_anchors.csv')
    >>> get_anchors(path_csv).tolist()  # doctest: +NORMALIZE_WHITESPACE
    [[10.0, 13.0], [16.0, 30.0], [33.0, 23.0], [30.0, 61.0], [62.0, 45.0], [59.0, 119.0],
     [116.0, 90.0], [156.0, 198.0], [373.0, 326.0]]
    """
    assert os.path.isfile(path_anchors), 'missing file: %s' % path_anchors
    df = pd.read_csv(path_anchors, header=None, index_col=None)
    anchors = df.values.astype(float)
    return anchors


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value


    Example
    -------
    >>> bboxes = [[100, 150, 200, 250, 0], [300, 50, 400, 200, 1]]
    >>> anchors = get_anchors(os.path.join(update_path('model_data'), 'yolo_anchors.csv'))
    >>> anchors.shape
    (9, 2)
    >>> true_boxes = preprocess_true_boxes(np.array([bboxes]), (416, 416), anchors, 5)
    >>> len(true_boxes)
    3
    >>> true_boxes[0].shape
    (1, 13, 13, 3, 10)
    """
    assert (true_boxes[..., 4] < num_classes).all(), \
        'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] \
        if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    nb_boxes = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[layer_idx]
                   for layer_idx in range(num_layers)]
    y_true = [np.zeros((nb_boxes, grid_shapes[layer_idx][0], grid_shapes[layer_idx][1],
                        len(anchor_mask[layer_idx]), 5 + num_classes),
                       dtype='float32') for layer_idx in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for bi in range(nb_boxes):
        # Discard zero rows.
        wh = boxes_wh[bi, valid_mask[bi]]
        if len(wh) == 0:
            continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for layer_idx in range(num_layers):
                if n not in anchor_mask[layer_idx]:
                    continue
                i = np.floor(true_boxes[bi, t, 0] * grid_shapes[layer_idx][1]).astype('int32')
                j = np.floor(true_boxes[bi, t, 1] * grid_shapes[layer_idx][0]).astype('int32')
                k = anchor_mask[layer_idx].index(n)
                c = true_boxes[bi, t, 4].astype('int32')
                y_true[layer_idx][bi, j, i, k, 0:4] = true_boxes[bi, t, 0:4]
                y_true[layer_idx][bi, j, i, k, 4] = 1
                y_true[layer_idx][bi, j, i, k, 5 + c] = 1

    return y_true


def data_generator(annotation_lines, input_shape, anchors, nb_classes,
                   batch_size=1, augment=True, max_boxes=20,
                   jitter=0.3, img_scaling=1.2, resize_img=True, allow_rnd_shift=True,
                   color_hue=0.1, color_sat=1.5, color_val=1.5,
                   flip_horizontal=True, flip_vertical=False,
                   bbox_overlap=0.95, nb_threads=1):
    """data generator for fit_generator

    :param list(str) annotation_lines:
    :param int batch_size:
    :param ndarray anchors:
    :param int nb_classes:
    :param tuple(int,int) input_shape: CNN input size
    :param bool augment: perform augmentation
    :param int max_boxes: maximal number of training bounding boxes
    :param float jitter:
    :param float color_hue: range of change of HSV color HUE
    :param float color_sat: range of change of HSV color SAT
    :param float color_val: range of change of HSV color value
    :param float img_scaling: upper image scaling
    :param bool flip_horizontal: allow random flop image/boxes vertical
    :param bool flip_vertical: allow random flop image/boxes horizontal
    :param bool resize_img: resize image to fit fully to CNN
    :param bool allow_rnd_shift: allow shifting image not only centered crop
    :param float bbox_overlap: threshold in case cut image, drop all boxes with lower overlap
    :param float|int nb_threads: nb threads running in parallel
    :return:

    >>> np.random.seed(0)
    >>> path_img = os.path.join(update_path('model_data'), 'bike-car-dog.jpg')
    >>> line = path_img + ' 100,150,200,250,0 300,50,400,200,1'
    >>> anchors = get_anchors(os.path.join(update_path('model_data'), 'yolo_anchors.csv'))
    >>> gen = data_generator([line], (416, 416), anchors, 3, nb_threads=2)
    >>> batch = next(gen)
    >>> len(batch)
    2
    >>> [b.shape for b in batch[0]]
    [(1, 416, 416, 3), (1, 13, 13, 3, 8), (1, 26, 26, 3, 8), (1, 52, 52, 3, 8)]
    >>> gen = data_generator([line], (416, 416), anchors, 3, augment=False)
    >>> batch = next(gen)
    >>> len(batch)
    2
    >>> [b.shape for b in batch[0]]
    [(1, 416, 416, 3), (1, 13, 13, 3, 8), (1, 26, 26, 3, 8), (1, 52, 52, 3, 8)]
    """
    nb_lines = len(annotation_lines)
    circ_i = 0
    if nb_lines == 0 or batch_size <= 0:
        return None

    color_hue = abs(color_hue)
    color_sat = color_sat if color_sat > 1 else 1. / color_sat
    color_val = color_val if color_val > 1 else 1. / color_val

    nb_threads = nb_workers(nb_threads)
    pool = ProcessPool(nb_threads) if nb_threads > 1 else None
    _wrap_rand_data = partial(
        get_augmented_data,
        input_shape=input_shape,
        augment=augment,
        max_boxes=max_boxes,
        jitter=jitter,
        resize_img=resize_img,
        img_scaling=img_scaling,
        allow_rnd_shift=allow_rnd_shift,
        hue=color_hue,
        sat=color_sat,
        val=color_val,
        flip_horizontal=flip_horizontal,
        flip_vertical=flip_vertical,
        bbox_overlap=bbox_overlap,
    )

    while True:
        if circ_i < batch_size:
            # shuffle while you are starting new cycle
            np.random.shuffle(annotation_lines)
        batch_image_data = []
        batch_box_data = []

        # create the list of lines to be loaded in batch
        annot_lines = annotation_lines[circ_i:circ_i + batch_size]
        batch_offset = (circ_i + batch_size) - nb_lines
        # chekck if the loaded batch size have sufficient size
        if batch_offset > 0:
            annot_lines += annotation_lines[:batch_offset]
        # multiprocessing loading of batch data
        map_process = pool.imap if pool else map
        for image, box in map_process(_wrap_rand_data, annot_lines):
            batch_image_data.append(image)
            batch_box_data.append(box)

        circ_i = (circ_i + batch_size) % nb_lines

        batch_image_data = np.array(batch_image_data)
        batch_box_data = np.array(batch_box_data)
        y_true = preprocess_true_boxes(batch_box_data, input_shape, anchors, nb_classes)
        batch = [batch_image_data, *y_true], np.zeros(batch_size)
        yield batch
        gc.collect()

    if pool:
        pool.close()
        pool.join()
        pool.clear()


def generator_bottleneck(annotation_lines, batch_size, input_shape, anchors, nb_classes,
                         bottlenecks, randomize=False):
    n = len(annotation_lines)
    circ_i = 0
    while True:
        box_data = []
        b0 = np.zeros((batch_size, bottlenecks[0].shape[1],
                       bottlenecks[0].shape[2], bottlenecks[0].shape[3]))
        b1 = np.zeros((batch_size, bottlenecks[1].shape[1],
                       bottlenecks[1].shape[2], bottlenecks[1].shape[3]))
        b2 = np.zeros((batch_size, bottlenecks[2].shape[1],
                       bottlenecks[2].shape[2], bottlenecks[2].shape[3]))
        for b in range(batch_size):
            _, box = get_augmented_data(annotation_lines[circ_i], input_shape,
                                        augment=randomize, img_scaling=1.)
            box_data.append(box)
            b0[b] = bottlenecks[0][circ_i]
            b1[b] = bottlenecks[1][circ_i]
            b2[b] = bottlenecks[2][circ_i]
            circ_i = (circ_i + 1) % n
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, nb_classes)
        yield [b0, b1, b2, *y_true], np.zeros(batch_size)


def check_params_path(params):
    for k in (k for k in params if 'path' in k):
        if not params[k]:
            continue
        params[k] = update_path(params[k])
        assert os.path.exists(params[k]), 'missing (%s): %s' % (k, params[k])
    return params
