import tensorflow as tf


def diff(x, axis):
    """
    Calculate the discrete difference along the given axis.
    The first difference is given by ``out[i] = x[i+1] - x[i]`` along
    the given axis, higher differences are calculated by using `diff`
    recursively.
    Parameters
    ----------
    x : array_like
        Input array
    axis : int
        The axis along which the difference is taken, default is the last axis.
    Returns
    -------
    diff : ndarray
        The 1-th differences. The shape of the output is the same as `x`
        except along `axis` where the dimension is smaller by `n`. The
        type of the output is the same as the type of the difference
        between any two elements of `x`. This is the same as the type of
        `x` in most cases. A notable exception is `datetime64`, which
        results in a `timedelta64` output array.
    """
    shapex = tf.shape(x)
    xdiff = tf.roll(x, shift=-1, axis=axis) - x
    slice_start = tf.zeros(len(shapex), dtype=tf.int32)
    slice_end = shapex - tf.constant(
        [(i == axis) for i in range(len(shapex))], dtype=tf.int32
    )

    return tf.slice(xdiff, slice_start, slice_end)


def cubic_interp1d(x0, x, y):
    """
    Interpolate a 1-D function using cubic splines.
      x0 : a float or an 1d-array
      x : (N,) array_like
          A 1-D array of real/complex values.
      y : (N,) array_like
          A 1-D array of real values. The length of y along the
          interpolation axis must be equal to the length of x.

    Implement a trick to generate at first step the cholesky matrice L of
    the tridiagonal matrice A (thus L is a bidiagonal matrice that
    can be solved in two distinct loops).

    additional ref: www.math.uh.edu/~jingqiu/math4364/spline.pdf
    """
    size = len(x)

    xdiff = diff(x, axis=0)
    ydiff = diff(y, axis=0)
    # xdiff = tf.convert_to_tensor(np.diff(x))
    # ydiff = tf.convert_to_tensor(np.diff(y))
    # allocate buffer matrices
    Li = [0] * size
    Li_1 = [0] * (size - 1)
    z = [0] * size

    # fill diagonals Li and Li-1 and solve [L][y] = [B]
    Li[0] = tf.sqrt(2.0 * xdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0  # natural boundary
    z[0] = B0 / Li[0]

    for i in range(1, size - 1, 1):
        Li_1[i] = xdiff[i - 1] / Li[i - 1]
        Li[i] = tf.sqrt(2 * (xdiff[i - 1] + xdiff[i]) - Li_1[i - 1] * Li_1[i - 1])
        Bi = 6.0 * (ydiff[i] / xdiff[i] - ydiff[i - 1] / xdiff[i - 1])
        z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

    i = size - 1
    Li_1[i - 1] = xdiff[-1] / Li[i - 1]
    Li[i] = tf.sqrt(2 * xdiff[-1] - Li_1[i - 1] * Li_1[i - 1])
    Bi = 0.0  # natural boundary
    z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

    # solve [L.T][x] = [y]
    i = size - 1
    z[i] = z[i] / Li[i]
    for i in range(size - 2, -1, -1):
        z[i] = (z[i] - Li_1[i - 1] * z[i + 1]) / Li[i]

    # find index
    index = tf.searchsorted(x, x0)
    index = tf.clip_by_value(index, 1, size - 1)

    xi1, xi0 = tf.gather(x, index), tf.gather(x, index - 1)
    yi1, yi0 = tf.gather(y, index), tf.gather(y, index - 1)
    zi1, zi0 = tf.gather(z, index), tf.gather(z, index - 1)
    hi1 = xi1 - xi0

    # calculate cubic
    f0 = (
        zi0 / (6 * hi1) * (xi1 - x0) ** 3
        + zi1 / (6 * hi1) * (x0 - xi0) ** 3
        + (yi1 / hi1 - zi1 * hi1 / 6) * (x0 - xi0)
        + (yi0 / hi1 - zi0 * hi1 / 6) * (xi1 - x0)
    )
    return f0
