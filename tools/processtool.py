from math import sqrt

import cv2
import numpy as np


def mask_by_color(img, *args):
    """
    Generate masks for given color ranges and return the merged mask.

    Args:
        img: Input image in BGR format.
        *args: Color ranges in the format of tuples or lists.

    Returns:
        Merged mask containing all color ranges.
    """

    if not args:
        raise ValueError("At least one color range is required")

    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except Exception:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ranges_list = []

    # check if each of args values are tuple (convert each to list) or list (save as it is)
    # or raise TypeError
    for arg in args:
        if isinstance(arg, tuple):
            arg = list(arg)
        elif not isinstance(arg, list):
            raise TypeError(f"Invalid color range type: {type(arg)}")

        ranges_list.append(
            np.array(arg)
        )  # Convert to [[np-low, np-high], [np-low, np-high] ...]

    masks_list = []

    # create masks for each color range
    for ranges in ranges_list:
        mask = cv2.inRange(
            img, ranges[0], ranges[1]
        )  # create mask for each color range
        masks_list.append(mask)  # append mask to masks_list

    # merge all masks
    masks_merged = masks_list[0]  # first mask
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
    if wd and ht:
        divk = None
        center_area_height = ht
        center_area_width = wd
    else:
        center_area_height = round(ih / divk)
        center_area_width = round(iw / divk)
    top_offset = (ih - center_area_height) // 2
    left_offset = (iw - center_area_width) // 2
    print(center_area_height, center_area_width)
    cmask[
        top_offset : top_offset + center_area_height,
        left_offset : left_offset + center_area_width,
    ] = (255, 255, 255)
    return cmask


def get_dist_to_center(a, b):
    """
    Calculate the distance from the point (a, b) to the center (0, 0) in a 2D plane.

    Args:
        a (float): The x-coordinate of the point.
        b (float): The y-coordinate of the point.

    Returns:
        float: The distance from the point to the center, rounded to 2 decimal places.
    """
    a, b = abs(a), abs(b)
    return int(round(sqrt(a**2 + b**2)))
