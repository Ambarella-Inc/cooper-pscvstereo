
import numpy as np
from PIL import Image
from os.path import *
import re
import imageio
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            return np.resize(data, (int(h), int(w), 2))


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def writePFM(file, array):
    import os
    assert type(file) is str and type(array) is np.ndarray and \
           os.path.splitext(file)[1] == ".pfm"
    with open(file, 'wb') as f:
        H, W = array.shape
        headers = ["Pf\n", f"{W} {H}\n", "-1\n"]
        for header in headers:
            f.write(str.encode(header))
        array = np.flip(array, axis=0).astype(np.float32)
        f.write(array.tobytes())


def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    tmp = np.zeros((height, width*nBands))
    tmp[:, np.arange(width)*2] = u
    tmp[:, np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid


def readDispInStereo2K(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 100.0
    valid = disp > 0.0
    return disp, valid


def readDispVKITTI(filename):
    depth = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) * (1.0/ 100.0)
    disp = 0.532725 * 725.0087 / depth

    valid = (depth > 6.0) & (depth < 655.0)
    disp[depth == 0.0] = 0
    disp[depth > 655] = 0

    return disp, valid


# Method taken from https://github.com/castacks/tartanair_tools/blob/master/data_type.md
def readDispTartanAir(file_name):
    depth = np.load(file_name)
    disp = 80.0 / depth
    valid = disp > 0
    return disp, valid


def readDispCustomDataset(file_name):
    disp = np.load(file_name)
    nocc_pix = file_name.replace('disp.npy', 'mask.png')
    nocc_pix = imageio.imread(nocc_pix) == 255
    return disp, nocc_pix


def readDispMiddlebury(file_name):
    disp_name = basename(file_name)
    if disp_name in ('disp1.png', 'disp5.png'):
        # Middlebury 2005/2006 FullSize disparities are stored directly in
        # pixel units. A value of zero denotes unknown disparity.
        disp = imageio.imread(file_name).astype(np.float32)
        valid = disp > 0
        return disp, valid
    if disp_name == 'disp0GT.pfm':
        disp = readPFM(file_name).astype(np.float32)
        assert len(disp.shape) == 2
        nocc_pix = file_name.replace('disp0GT.pfm', 'mask0nocc.png')
        assert exists(nocc_pix)
        nocc_pix = imageio.imread(nocc_pix) == 255
        assert np.any(nocc_pix)
        nocc_pix = disp < 1e3
        return disp, nocc_pix
    elif disp_name == 'disp0.pfm':
        disp = readPFM(file_name).astype(np.float32)
        valid = disp < 1e3
        return disp, valid
    elif disp_name == 'disp1.pfm':
        disp = readPFM(file_name).astype(np.float32)
        valid = disp < 1e3
        return disp, valid


def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])


def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        img = Image.open(file_name)

        img = np.array(img).astype(np.uint8)
        if len(img.shape) > 2 and img.shape[-1] > 3:
            img = img[:, :3]
        img = Image.fromarray(img)
        return img
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]

    return []
