from math import sqrt

import cv2
import numpy as np


def mask_by_color(img, ranges):
    """
    Generate a mask for an image based on color ranges.

    Parameters:
    - img (numpy.ndarray): The image to mask
    - ranges (list/tuple/numpy.ndarray): Color ranges for masking

    Returns:
    - numpy.ndarray: The resulting mask

    This function converts the image to HSV for color masking. Supports both color and grayscale images.
    """
    if not isinstance(ranges, list):
        ranges = list(ranges)

    if not any(isinstance(x, list) for x in ranges):
        ranges = [ranges]

    if len(img.shape) != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ranges_list = [list(map(np.array, rang)) for rang in ranges]

    masks_list = []
    for ranges in ranges_list:
        mask = cv2.inRange(
            img,
            ranges[0],
            ranges[1],
        )
        masks_list.append(mask)

    masks_merged = masks_list[0]
    if len(masks_list) > 1:
        for i in range(1, len(masks_list)):
            masks_merged = cv2.bitwise_or(masks_merged, masks_list[i])

    return masks_merged


def mask_center(img, wd=None, ht=None, divk=2):
    """
    Create a center mask for the input image with the specified width and height, or a division factor.
    Parameters:
        img: input image
        wd (optional): width of the center area
        ht (optional): height of the center area
        divk (optional): division factor for determining the center area size, default is 2
    Returns:
        cmask: center mask image
    """
    ih, iw = img.shape[:2]
    cmask = np.zeros((ih, iw, 3), dtype=np.uint8)

    if wd is not None and ht is not None:
        divk = None
        center_area_height = ht
        center_area_width = wd
    else:
        center_area_height = ih // divk
        center_area_width = iw // divk

    top_offset = (ih - center_area_height) // 2
    left_offset = (iw - center_area_width) // 2

    cmask[
        slice(top_offset, top_offset + center_area_height),
        slice(left_offset, left_offset + center_area_width),
    ] = (255, 255, 255)

    return cmask


def invert(img):
    return 255 - img


def get_dist_to_center(a, b) -> float:
    """
    Calculate the distance from the point (a, b) to the center (0, 0) in a 2D plane.

    Args:
        a (float): The x-coordinate of the point.
        b (float): The y-coordinate of the point.

    Returns:
        float: The distance from the point to the center, rounded to 2 decimal places.
    """
    a, b = abs(a), abs(b)
    return round(sqrt(a ** 2 + b ** 2), 2)
