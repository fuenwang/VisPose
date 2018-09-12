from __future__ import print_function
import os
import sys
import cv2
import math
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pts = np.array(
    [
        [-1, -1, 2],
        [0, 0, 0],
        [-1, 1, 2],
        [0, 0, 0],
        [1, 1, 2],
        [0, 0, 0],
        [1, -1, 2],

        [1, 1, 2],
        [-1, 1, 2],
        [-1, -1, 2],
        [1, -1, 2],
        [0, -1, 2],
        [0, -0.5, 2]
    ]
)

# def euler2mat(z=0, y=0, x=0, isRadian=True):


def euler2mat(xyz, isRadian=True):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    if not isRadian:
        z = ((np.pi)/180.) * z
        y = ((np.pi)/180.) * y
        x = ((np.pi)/180.) * x
    assert z >= (-np.pi) and z < np.pi, 'Inapprorpriate z: %f' % z
    assert y >= (-np.pi) and y < np.pi, 'Inapprorpriate y: %f' % y
    assert x >= (-np.pi) and x < np.pi, 'Inapprorpriate x: %f' % x

    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
            [[cosy, 0, siny],
             [0, 1, 0],
             [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def mat2euler(M, cy_thresh=None, seq='zyx'):
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if seq == 'zyx':
        if cy > cy_thresh:  # cos(y) not close to zero, standard form
            z = math.atan2(-r12,  r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = math.atan2(r13,  cy)  # atan2(sin(y), cy)
            x = math.atan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = math.atan2(r21,  r22)
            y = math.atan2(r13,  cy)  # atan2(sin(y), cy)
            x = 0.0
    elif seq == 'xyz':
        if cy > cy_thresh:
            y = math.atan2(-r31, cy)
            x = math.atan2(r32, r33)
            z = math.atan2(r21, r11)
        else:
            z = 0.0
            if r31 < 0:
                y = np.pi/2
                x = atan2(r12, r13)
            else:
                y = -np.pi/2
                # x =
    else:
        raise Exception('Sequence not recognized')
    return np.array([x, y, z])


class Pose:
    def __init__(self, r, t, mode='euler'):
        self.r = np.asarray(r)
        self.t = np.asarray(t)
        if mode == 'euler':
            self.R = euler2mat(self.r / 180.0 * np.pi)
        elif mode == 'axis':
            self.R = cv2.Rodrigues(self.r)[0]
        else:
            print ('error')
            exit()
        self.T = self.t.copy()

    def points(self):
        tmp = []
        for i in range(pts.shape[0]):
            p = pts[i, :]
            new_p = self.R.T.dot(p - self.T).reshape([1, 3])
            tmp.append(new_p)
        tmp = np.concatenate(tmp, axis=0)
        #print (tmp.shape)
        return tmp

    def inv(self):
        tmp = Pose([0, 0, 0], [0, 0, 0])
        tmp.R = self.R.T
        tmp.r = list(mat2euler(tmp.R)/np.pi * 180)
        tmp.T = np.dot(-self.R.T, self.T)
        return tmp

    def __call__(self, pt):
        return np.asarray(pt).dot(self.R.T) + self.T
    
    def __repr__(self):
        return 'R: [%.2f %.2f %.2f], T: [%.2f %.2f %.2f]'%(self.r[0], self.r[1], self.r[2], self.T[0], self.T[1], self.T[2])


def Sequence(lst):
    tmp = Pose([0, 0, 0], [0, 0, 0])
    tmp.R = lst[0].R
    tmp.T = lst[0].T
    for i in range(1, len(lst)):
        now = lst[i]
        tmp.R = np.dot(now.R, tmp.R)
        tmp.T = np.dot(now.R, tmp.T) + now.T
    tmp.r = list(mat2euler(tmp.R)/np.pi * 180)
    return tmp


def init(ax):
    ax.plot([0, 1], [0, 0], [0, 0], 'r')
    ax.plot([0, 0], [0, 1], [0, 0], 'g')
    ax.plot([0, 0], [0, 0], [0, 1], 'b')

class PosePlt:
    def __init__(self, ax):
        self.ax = ax

    def plot(self, pose, color=None):
        pt = np.asarray(pose.points())
        if color is None:
            self.ax.plot(pt[:, 0], pt[:, 1], pt[:, 2])
        else:
            self.ax.plot(pt[:, 0], pt[:, 1], pt[:, 2], color)
    
    def nplot(self, lst):
        for i in lst:
            pt = np.asarray(i.points())
            self.ax.plot(pt[:, 0], pt[:, 1], pt[:, 2])