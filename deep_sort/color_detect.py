import cv2
import numpy as np

def find_color(roi, threshold=0.0):
  """
  This function is used to calculate the color ratio of the patch of image provided
  as the region of interest (roi) parameter.
  Returns the ratio of non white pixels to All pixels, lighter coloured jersies should have low value.
  """

  roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
  # set a min and max for team colors
  COLOR_MIN = np.array([0, 0, 0])
  COLOR_MAX = np.array([255, 255, 100])

  # # dark teams will remain with this mask
  mask = cv2.inRange(roi_hsv, COLOR_MIN, COLOR_MAX)
  res = cv2.bitwise_and(roi,roi, mask= mask)

  res = cv2.bitwise_and(roi,roi, mask= mask)

  # dark teams should have a higher ratio
  tot_pix = roi.any(axis=-1).sum()
  color_pix = res.any(axis=-1).sum()
  ratio = color_pix/tot_pix

  return(ratio)