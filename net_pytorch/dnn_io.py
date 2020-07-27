import numpy as np

def r2c(x, axis=1):
    """Convert pseudo-complex data (2 real channels) to complex data
    x: ndarray   input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm) """
    shape = x.shape
    if axis < 0: axis = x.ndim + axis
    ctype = np.complex64 if x.dtype == np.float32 else np.complex128
    # ctype = np.complex64
    if axis < len(shape):
        newshape = tuple([i for i in range(0, axis)]) + tuple([i for i in range(axis+1, x.ndim)]) + (axis,)
        x = x.transpose(newshape)
    x = np.ascontiguousarray(x).view(dtype=ctype)
    return x.reshape(x.shape[:-1])

def c2r(x, axis=1):
    """Convert complex data to pseudo-complex data (2 real channels)
    x: ndarray   input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)  """
    shape = x.shape #ori
    dtype = np.float32 if x.dtype == np.complex64 else np.float64
    # dtype = np.float16
    x = np.ascontiguousarray(x).view(dtype=dtype).reshape(shape + (2,)) 
    n = x.ndim
    if axis < 0: axis = n + axis
    if axis < n:
        newshape = tuple([i for i in range(0, axis)]) + (n-1,) + tuple([i for i in range(axis, n-1)])
        x = x.transpose(newshape)
    return x

def to_tensor_format(x):
    """ Assumes data is of shape (n[, nt], nx, ny). Reshapes to (n, n_channels, nx, ny[, nt])
    Note: Depth must be the last axis, the dimensions will be reordered  """
    if x.ndim == 4:  
        x = np.transpose(x, (0, 2, 3, 1))
    x = c2r(x)
    return x

def from_tensor_format(x):
    """ Assumes data is of shape (n, 2, nx, ny[, nt]).  Reshapes to (n, [nt, ]nx, ny) """
    if x.ndim == 5:  
        x = np.transpose(x, (0, 2, 1, 3, 4))
    x = r2c(x)
    return x