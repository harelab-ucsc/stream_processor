#!/usr/bin/env python3

from scipy.spatial.transform import Rotation as R
import numpy as np

# import sys
# import cv2

# import apriltag


def dimIterProd(mtx_dims):
    tmp = 1
    for d in mtx_dims:
        tmp *= d
    return tmp


def array_flatten(mtx, mtx_dims):
    """mtx is m x n (rectangular, nonsparse) array"""
    val = []
    for i in range(mtx_dims[0]):
        for j in range(mtx_dims[1]):
            val.append(mtx[i][j])
    return val


def array_expand(mtx, mtx_dims):
    """mtx is mn x 1 (m*n dimensional vector)"""
    tmp = []
    tmps = []
    row = 0
    for i, elem in enumerate(mtx):
        tmp.append(elem)
        if i == row * mtx_dims[1] + (mtx_dims[1] - 1):
            row += 1
            tmps.append(tmp)
            tmp = []
    return tmps


def matrix_list_converter(mtx: list, mtx_dims):
    if len(mtx) == dimIterProd(mtx_dims):
        return array_expand(mtx, mtx_dims)
    else:
        return array_flatten(mtx, mtx_dims)


def string_list_converter(foo):
    if isinstance(foo, str):
        if foo != "None":
            val = []
            tmp = foo.split("[")[1]
            # print(tmp)
            tmp = tmp.split("]")[0]
            # print(tmp)
            for item in tmp.split(", "):
                if item != "":
                    val.append(float(item))
            # print(val)
            return val
        else:
            return None
    elif isinstance(foo, list):
        val = "["
        for item in foo:
            val += str(item)
        val += "]"
        return val
    else:
        print("oops")


def poseRowToTransform(pose):
    """Given a row from the db, produce a 4x4 homogeneous transform
    Return as a 4x4 nparray"""
    rot = R.from_quat(pose[3:7]).as_matrix()
    t = np.array([pose[0], pose[1], pose[2]])
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = t
    return T
