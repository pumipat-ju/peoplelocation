"""Pure topology gate primitives."""
import cv2
import numpy as np

def point_in_polygon(point, polygon_pts):
    if polygon_pts is None:
        return True
    poly = np.array(polygon_pts, dtype=np.int32)
    return cv2.pointPolygonTest(poly, point, False) >= 0
