from __future__ import division
import numpy as np
from scipy.linalg import toeplitz, expm
from scipy.stats import norm, binom
from math import sqrt
from scipy.integrate import quad
from scipy import optimize

def kronn(*args):
    # returns multidimensional kronecker product of all matrices in the argument list
    z = args[0]
    for i in range(1, len(args)):
        z = np.kron(z, args[i])
    return z


def beliefTransitionMatrix(p_appear, p_disappear, nq, w):
    """
    create transition matrix between nq belief states q to q'
    :param p_appear:
    :param p_disappear:
    :param nq:
    :param w: width of diffusive noise
    :return:
    """
    Tqqq = np.zeros((nq, nq))
    dq = 1 / nq
    a = 1 - p_disappear - p_appear

    for i in range(nq):
        for j in range(nq):
            q = i * dq
            qq = j * dq

            bm = (qq - p_appear) / a
            bp = (qq + dq - p_appear) / a

            Tqqq[j, i] = max(0, min(q + dq, bp) - max(q, bm) )
            Tqqq[j, i] = Tqqq[j, i] / (bp - bm) * sqrt(dq ** 2 + (bp - bm) ** 2)
    Tqqq = Tqqq / np.tile(np.sum(Tqqq, 0), (nq, 1))

    nt = 20
    d = w / nt
    dD = toeplitz(np.insert(np.zeros(nq - 2), 0, np.array([-2 * d, d])))
    dD[1, 0] = 2 * d
    dD[-2, -1] = 2 * d
    D = expm(dD * nt)
    D = D / np.tile(np.sum(D, 0), (nq, 1))

    Tqqq = np.dot(D, Tqqq)

    return Tqqq


def beliefTransitionMatrixGaussian(p_appear, p_disappear, nq, sigma = 0.1):
    mu = 0

    d = np.zeros((nq, nq))
    Trrr = np.zeros((nq, nq))
    dq = 1 / nq
    a = 1 - p_disappear - p_appear

    for i in range(nq):
        for j in range(nq):
            q = i * dq + dq / 2
            qq = j * dq + dq / 2

            d[j, i] = abs(a * q - qq + p_appear) / sqrt(a ** 2 + 1)
            Trrr[j, i] = norm.pdf(d[j, i], mu, sigma)

    Tb = Trrr / np.tile(np.sum(Trrr, 0), (nq, 1))

    return Tb

def beliefTransitionMatrixGaussianCol(p_appear, p_disappear, qmin, qmax, Ncol, nq):
    def gb(x, k1, k0, p_appear, p_disappear):
        a = 1 - p_disappear - p_appear
        return (k1 * a * x + k1 * p_appear) / ((k1 - k0) * a * x + k1 * p_appear + k0 * (1 - p_appear))

    def gbinv(y, k1, k0, p_appear, p_disappear):
        a = 1 - p_disappear - p_appear
        return (y * (k1 * p_appear + k0 * (1 - p_appear)) - k1 * p_appear) / (k1 * a - y * (k1 - k0) * a)

    mu = 0

    Trans_state = np.array([[1 - p_appear, p_disappear],
                            [p_appear, 1 - p_disappear]])
    Obs_emis = np.zeros((Ncol + 1, 2))  # Observation(color) generation
    Obs_emis[:, 0] = binom.pmf(range(Ncol + 1), Ncol, qmax)
    Obs_emis[:, 1] = binom.pmf(range(Ncol + 1), Ncol, qmin)

    dq = 1 / nq

    d = np.zeros((Ncol + 1, nq, nq))
    den = np.zeros((Ncol + 1, nq, nq))
    xopt = np.zeros((Ncol + 1, nq, nq))
    height = np.zeros((Ncol + 1, nq, nq))
    Trans_belief_obs_approx = np.zeros((Ncol + 1, nq, nq))

    for n in range(Ncol + 1):
        k0 = Obs_emis[n, 0]
        k1 = Obs_emis[n, 1]

        for i in range(nq):
            for j in range(nq):
                # xmin = dq * i
                # xmax = dq * (i + 1)
                # ymin = dq * j
                # ymax = dq * (j + 1)
                #
                # bl = max(gbinv(ymin, k1, k0, p_appear, p_disappear), xmin)
                # br = min(gbinv(ymax, k1, k0, p_appear, p_disappear), xmax)
                #
                # if bl > br:
                #     Trans_belief_obs[n, j, i] = 0
                # else:
                #     Trans_belief_obs[n, j, i] = \
                #     quad(lambda x: Obs_emis[n, :].dot(Trans_state).dot(np.array([1 - x, x])),
                #         bl, br)[0]

                # Approximate the probability with Gaussian approxiamtion
                q = i * dq + dq / 2
                qq = j * dq + dq / 2

                def dist(x):
                    return sqrt((q - x) ** 2 + (qq - gb(x, k1, k0, p_appear, p_disappear)) ** 2)

                xopt[n, j, i], d[n, j, i] = optimize.fminbound(dist, 0, 1, full_output=1)[0:2]
                den[n, j, i] = norm.pdf(d[n, j, i], mu, 1 / nq / 3)   # use this to approximate delta function with diffusion

                height[n, j, i] = Obs_emis[n, :].dot(Trans_state).dot(np.array([1 - xopt[n, j, i], xopt[n, j, i]]))

        den[n] = den[n] / np.tile(np.sum(den[n], 0), (nq, 1))
        Trans_belief_obs_approx[n] = np.multiply(den[n], height[n])

    return Trans_belief_obs_approx, Obs_emis.dot(Trans_state), den


def beliefTransitionMatrixGaussianDerivative(p_appear, p_disappear, nq, sigma=0.1):
    mu = 0

    d = np.zeros((nq, nq))
    pdpgamma = np.zeros((nq, nq))  # derivative with respect to the appear rate, p_appear
    pdpepsilon = np.zeros((nq, nq)) # derivative with respect to the disappear rate, p_disappear
    dq = 1 / nq
    a = 1 - p_disappear - p_appear

    for i in range(nq):
        for j in range(nq):
            q = i * dq + dq / 2
            qq = j * dq + dq / 2

            d[j, i] = abs(a * q - qq + p_appear) / sqrt(a ** 2 + 1)
            pdpepsilon[j, i] = np.sign(a * q - qq + p_appear) * \
                               (-q * sqrt(a ** 2 + 1) + a/sqrt(a ** 2 + 1) * (a * q - qq + p_appear)) / (a ** 2 + 1)
            pdpgamma[j, i] = np.sign(a * q - qq + p_appear) * \
                             ((1-q) * sqrt(a ** 2 + 1) + a/sqrt(a ** 2 + 1) * (a * q - qq + p_appear)) / (a ** 2 + 1)

    Trrr = norm.pdf(d, mu, sigma)  # probability from Gaussian distribution
    pTrrrpd = Trrr * d * (-1) / (sigma ** 2)  # partial derivative of Trrr with respect to d

    dTbdgamma = np.zeros((nq, nq))  # derivative of Tb with respect to the p_appear(gamma) rate
    dTbdepsilon = np.zeros((nq, nq))  # derivative of Tb with respect to the p_disappear(epsilon) rate

    for i in range(nq):
        for j in range(nq):
            dTbdgamma[j, i] = 1 / np.sum(Trrr[:, i]) * pTrrrpd[j, i] * pdpgamma[j, i] - \
                             Trrr[j, i] / (np.sum(Trrr[:, i]) ** 2) * np.sum(pTrrrpd[:, i] * pdpgamma[:, i])
            dTbdepsilon[j, i] = 1 / np.sum(Trrr[:, i]) * pTrrrpd[j, i] * pdpepsilon[j, i] - \
                             Trrr[j, i] / (np.sum(Trrr[:, i]) ** 2) * np.sum(pTrrrpd[:, i] * pdpepsilon[:, i])
    Tb = Trrr / np.tile(np.sum(Trrr, 0), (nq, 1))

    return dTbdgamma, dTbdepsilon


'''
##############################################################################################
The im2col and col2im functions referred to the code on
http://fhs1.bitbucket.org/glasslab_cluster/_modules/glasslab_cluster/utils.html
'''
''' (My algorithm)
def _im2col_distinct(A, size):
    dx, dy = size
    assert A.shape[0] % dx == 0
    assert A.shape[1] % dy == 0

    ncol = (A.shape[0]//dx) * (A.shape[1]//dy)
    R = np.empty((ncol, dx * dy), dtype = A.dtype)
    k = 0
    for j in range(0, A.shape[1], dy):
        for i in range(0, A.shape[0], dx):
            R[k, :] = A[i : i + dx, j : j + dy].ravel(order = 'F')
            k += 1
    return R.T

def _im2col_sliding(A, size):
    dx, dy = size
    xsz = A.shape[0] - dx + 1
    ysz = A.shape[1] - dy + 1
    R = np.empty((xsz * ysz, dx * dy), dtype = A.dtype)

    for j in range(ysz):
        for i in range(xsz):
            R[j * xsz + i, :] = A[i : i + dx, j : j + dy].ravel(order = 'F')
    return R.T
'''


def _im2col_distinct(A, size):
    A = A.T
    dy, dx = size[::-1]
    assert A.shape[0] % dy == 0
    assert A.shape[1] % dx == 0

    ncol = (A.shape[0]//dy) * (A.shape[1]//dx)
    R = np.empty((ncol, dx*dy), dtype=A.dtype)
    k = 0
    for i in range(0, A.shape[0], dy):
        for j in range(0, A.shape[1], dx):
            R[k, :] = A[i:i+dy, j:j+dx].ravel()
            k += 1
    return R.T


def _im2col_sliding(A, size):
    A = A.T
    dy, dx = size[::-1]
    xsz = A.shape[1]-dx+1
    ysz = A.shape[0]-dy+1
    R = np.empty((xsz*ysz, dx*dy), dtype=A.dtype)

    for i in range(ysz):
        for j in range(xsz):
            R[i*xsz+j, :] = A[i:i+dy, j:j+dx].ravel()
    return R.T


def im2col(A, size, type='sliding'):
    """This function behaves similar to *im2col* in MATLAB.

    Parameters
    ----------
    A : 2-D ndarray
        Image from which windows are obtained.
    size : 2-tuple
        Shape of each window.
    type : {'sliding', 'distinct'}, optional
        The type of the windows.

    Returns
    -------
    windows : 2-D ndarray
        The flattened windows stacked vertically.

    """

    if type == 'sliding':
        return _im2col_sliding(A, size)
    elif type == 'distinct':
        return _im2col_distinct(A, size)
    raise ValueError("invalid type of window")


def _col2im_distinct(R, size, width):
    R = R.T
    dy, dx = size[::-1]

    assert width % dx == 0
    nwinx = width//dx
    xsz = nwinx*dx

    assert R.shape[0] % nwinx == 0
    nwiny = R.shape[0]//nwinx
    ysz = nwiny*dy

    A = np.empty((ysz, xsz), dtype=R.dtype)
    k = 0
    for i in range(0, ysz, dy):
        for j in range(0, xsz, dx):
            A[i:i+dy, j:j+dx] = R[k].reshape(size[::-1])
            k += 1
    return A.T


def _col2im_sliding(R, size, width):
    '*********** This is not the same in Matlab, need to be modified later *****************'
    R = R.T
    dy, dx = size
    xsz = width-dx+1
    ysz = R.shape[0]//xsz

    A = np.empty((ysz+(dy-1), width), dtype = R.dtype)
    for i in range(ysz):
        for j in range(xsz):
            A[i:i+dy, j:j+dx] = R[i*xsz+j, :].reshape(dy, dx)
    return A.T


def col2im(R, size, width, type='sliding'):
    """This function behaves similar to *col2im* in MATLAB.

    It is the inverse of :func:`im2col`::

            A == col2im(im2col(A, size), size, A.shape[1])

    `R` is what `im2col` returns. `Size` and `type` are the same as
    in `im2col`. `Width` is the number of columns in `A`.

    Examples
    --------
    import numpy as np
    a = np.arange(12).reshape(3,4)
    a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    b = im2col(a, (2,2))
    b
    array([[ 0,  1,  4,  5],
           [ 1,  2,  5,  6],
           [ 2,  3,  6,  7],
           [ 4,  5,  8,  9],
           [ 5,  6,  9, 10],
           [ 6,  7, 10, 11]])
    col2im(b, (2,2), a.shape[1])
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])

    """
    if type == 'sliding':
        return _col2im_sliding(R, size, width)
    elif type == 'distinct':
        return _col2im_distinct(R, size, width)
    raise ValueError("invalid type of window")

'''
The im2col and col2im functions referred to the code on
http://fhs1.bitbucket.org/glasslab_cluster/_modules/glasslab_cluster/utils.html
######################################################################################################
'''


def reversekron(AB, n):
    BA = col2im(im2col(AB, tuple(n[1] * np.array([1, 1])), 'distinct').T, tuple(n[0] * np.array([1, 1])),
         np.prod(n), 'distinct')
    return BA


def tensorsum(A, B):
    ra, ca = A.shape
    rb, cb = B.shape
    C = np.empty((ra * rb, ca * cb))

    for i in range(ra):
        for j in range(ca):
            C[i*rb : (i+1)*rb, j*cb : (j+1)*cb] = A[i, j] + B

    return C


def tensorsumm(*args):
    '''
    :param args:
    :return: returns multidimensional kronecker sum of all matrices in list
    Note that the args must be two-dimensional. When any of the ars is a vector, need to pass in a
i    '''
    z = args[0]
    for i in range(1, len(args)):
        z = tensorsum(z, args[i])

    return z


def softmax(x, t):
    # transform the value of a vector x to softmax
    # beta is the te,perature parameter
    z = np.exp(x/t)
    z = z / np.max(z)
    return z / np.sum(z)



def QfromV(ValueIteration):
    Q = np.zeros((ValueIteration.A, ValueIteration.S))
    for a in range(ValueIteration.A):
        Q[a, :] = ValueIteration.R[a] + ValueIteration.discount * \
                                        ValueIteration.P[a].dot(ValueIteration.V)

    return Q


def QfromV_pi(PolicyIteration):
    Q = np.zeros((PolicyIteration.A, PolicyIteration.S))
    for a in range(PolicyIteration.A):
        Q[a, :] = PolicyIteration.R[a] + PolicyIteration.discount * \
                                        PolicyIteration.P[a].dot(PolicyIteration.V)
    return Q
