import numpy as np
import cv2


def _pad_width_center(w, target_w):
    left = (target_w - w) // 2
    right = target_w - w - left
    return left, right


def _pad_width_right(w, target_w):
    return 0, target_w - w


def _pad_height_center(h, target_h):
    top = (target_h - h) // 2
    bottom = target_h - h - top
    return top, bottom


def _pad_height_bottom(h, target_h):
    return 0, target_h - h


def VStack(*imgs, align='center'):
    max_w = max([_.shape[1] for _ in imgs])
    imgs_padded = []

    if align == 'center':
        for img in imgs:
            left, right = _pad_width_center(img.shape[1], max_w)
            imgs_padded.append(cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT))

    elif align == 'left':
        for img in imgs:
            left, right = _pad_width_right(img.shape[1], max_w)
            imgs_padded.append(cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT))

    else:
        raise ValueError('Unsupported alignment %s' % align)

    return np.concatenate(imgs_padded, axis=0)


def HStack(*imgs, align='center'):
    max_h = max([_.shape[0] for _ in imgs])

    imgs_padded = []

    if align == 'center':
        for img in imgs:
            top, bottom = _pad_height_center(img.shape[0], max_h)
            imgs_padded.append(cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT))

    elif align == 'top':
        for img in imgs:
            top, bottom = _pad_height_bottom(img.shape[0], max_h)
            imgs_padded.append(cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT))

    else:
        raise ValueError('Unsupported alignment %s' % align)

    return np.concatenate(imgs_padded, axis=1)


def Grid(*imgs, n_col=1, align='center'):
    chunks = [imgs[i:i + n_col] for i in range(0, len(imgs), n_col)]
    row_imgs = [HStack(*_, align=align) for _ in chunks]
    return VStack(*row_imgs, align=align)
