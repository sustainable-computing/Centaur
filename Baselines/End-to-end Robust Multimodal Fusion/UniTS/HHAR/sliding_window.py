
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

def normalize_shape(shape):
    '''
    This function normalizes array shapes and they are returned as a tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # if shape not a number
        pass
    try:
        t = tuple(shape)
        return t
    except TypeError:
        # if shape not iterable
        pass

    raise TypeError('shape must be an integer, or a tuple of integers')

def sliding_window(arr,win,sl = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
    Parameters:
        arr  - n-D numpy array
        win - an int (if arr is 1D) or tuple (if arr is 2D or more) which represents the size of each dimension of the window
        sl - an int (if arr is 1D) or tuple (if arr is 2D or more) represents the sliding length (in each dimension). 
        flatten - if True, all slices are flattened, otherwise, there is extra dimension for each dimension of the input.
    '''

    if None is sl:
        # sl not provided. the windows not overlap in any direction.
        sl = win
    win = normalize_shape(win)
    sl = normalize_shape(sl)

    # convert win, sl, and arr.shape to numpy arrays
    win = np.array(win)
    sl = np.array(sl)
    shape = np.array(arr.shape)

    # ensure that win, sl, and arr.shape all have the same number of dimensions
    ls = [len(shape),len(win),len(sl)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'arr.shape, win and sl all must have the same length. They were %s' % str(ls))

    if np.any(win > shape):
        raise ValueError(\
        'win should not be larger than arr in any dimension.\
 arr.shape was %s and win was %s' % (str(arr.shape),str(win)))

    n_shape = normalize_shape(((shape - win) // sl) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    n_shape += normalize_shape(win)
    newstr = normalize_shape(np.array(arr.strides) * sl) + arr.strides
    strided = ast(arr,shape = n_shape,strides = newstr)
    if not flatten:
        return strided

    m = len(win) if win.shape else 0
    fdim = (np.product(n_shape[:-m]),) if win.shape else ()
    dim = fdim + (n_shape[-m:])
    return strided.reshape(dim)