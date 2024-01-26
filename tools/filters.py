import cv2

def bright_contrast(input_img, brightness = 0, contrast = 0):
    """
    Apply brightness and contrast adjustments to an input image.

    Args:
        input_img: The input image to be processed.
        brightness: The brightness adjustment value. Defaults to 0.
        contrast: The contrast adjustment value. Defaults to 0.

    Returns:
        The processed image with brightness and contrast adjustments applied.
    """
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf



def histogram_adp(img, cllimit=1.0, gridsize=(6,6)):
    """
    Adaptive histogram equalization for brightness channel only (Y)

    Args:
        img: Input image.
        cllimit: Clip limit for contrast limiting.
        gridsize: Size of grid for histogram equalization.

    Returns:
        Equalized image.
    """

    # Adaptive histogram equalization
    # https://en.wikipedia.org/wiki/Adaptive_histogram_equalization

    clahe = cv2.createCLAHE(clipLimit=cllimit, tileGridSize=gridsize)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) # convert to YUV
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0]) # equalize the Y channel
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) # convert to BGR
    return img