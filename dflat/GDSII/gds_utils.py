import numpy as np
import gdspy
import cv2


def upsample_block(params, mask, cell_size, block_size):
    H, W, C = params.shape
    scale_factor = np.array(block_size) / np.array(cell_size)
    Hnew = np.rint(H * scale_factor[0]).astype(int)
    Wnew = np.rint(W * scale_factor[1]).astype(int)

    if C == 1:
        # Resize each channel separately then stack
        resized = cv2.resize(
            params[:, :, 0], (Wnew, Hnew), interpolation=cv2.INTER_LINEAR
        )
        params = resized[:, :, np.newaxis]  # Restore channel dim
    else:
        # Multi-channel image: OpenCV handles it fine
        params = cv2.resize(params, (Wnew, Hnew), interpolation=cv2.INTER_LINEAR)

    mask = cv2.resize(
        mask.astype(np.float16),
        (Wnew, Hnew),
        interpolation=cv2.INTER_NEAREST,
    )
    mask = mask.astype(bool)
    return params, mask


def add_marker_tag(cell, ms, hx, hy):
    mw = ms * 5 / 9
    mh = ms / 2 + (21.10 / ms)  # Correction for the + sign center

    cx = -mw / 2
    cy = -mh + 2
    # Add alignment markers at
    cell.add(gdspy.Text("+", ms, (cx - ms / 2, cy - ms / 2)))  # lower left
    cell.add(gdspy.Text("+", ms, (cx + hx + ms / 2, cy - ms / 2)))  # lower right
    cell.add(gdspy.Text("+", ms, (cx + hx + ms / 2, cy + hx + ms / 2)))  # upper right
    cell.add(gdspy.Text("+", ms, (cx - ms / 2, cy + hx + ms / 2)))  # upper left

    return
