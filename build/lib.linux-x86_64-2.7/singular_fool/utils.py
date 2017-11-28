from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator
import tensorflow as tf


def psi(x, p=2):
    """The function which is used to define the Boyd Power Method
    psi(x, p) = sign(x) * abs(x) ** (p - 1)
    entrywise.
    """
    return np.sign(x) * np.abs(x) ** (p - 1)


def get_gradients(x, fx):
    """Constructs required matvecs (implements alg. 3 from the paper)

    Args:
        x: tf.placeholder specifying the input of the DNN
        fx: tf.Tensor, output of the hidden layer to be used for constructing
        adversarial perturbation

    Returns:
        v - tf.placeholder for computing Jv
        w - tf.placeholder for computing J^T w
        J_v - tf.Tensor representing Jv
        Jt_w - tf.Tensor representing J^T w

    """
    input_shape = x.shape.as_list()[1:]
    hidden_shape = fx.shape.as_list()[1:]
    hidden_size = np.prod(hidden_shape)
    input_size = np.prod(input_shape)

    with tf.name_scope('matvecs'):
        w = tf.placeholder(tf.float32, shape=(hidden_size, ))
        v = tf.placeholder(tf.float32, shape=(input_size, ))
        fx_flat = tf.reshape(fx, (-1, hidden_size))
        h_dot_w = tf.tensordot(fx_flat, w, axes=[[1], [0]])
        Jtw_tensor = tf.gradients(h_dot_w[0], x)[0][0]
        Jt_w = tf.reshape(Jtw_tensor, (-1, ))
        Jtw_dot_v = tf.reduce_sum(Jt_w * v)
        J_v = tf.gradients(Jtw_dot_v, w)[0]
        return v, w, J_v, Jt_w


def get_jac_matvecs(x, v, w, jv, jtw, sess, img_batch, aux_tensors=None):
    """ Given a batch of images constructs full Jacobian matvecs (implements
    alg. 2 from the paper)

    Args:
        x: tf.placeholder specifying the input of the DNN
        v, w, jv, jtw: tf.Tensors as returned by get_gradients
        sess: tf.Session
        img_batch: numpy tensor of size (N, H, W, C) specifying batch of images
        used for the power method (as discussed in section 4)
        aux_tensors: optional, dictionary of tf.placeholders to feed to sess
        e.g. training/testing phase.

    Returns:
        jac - scipy.sparse.linalg.LinearOperator
    """
    batch_size = img_batch.shape[0]
    input_size = jtw.shape.as_list()[0]
    hidden_size = jv.shape.as_list()[0]

    def rmv(vec):
        res = np.zeros(input_size)
        vec = np.squeeze(vec)

        def grad_func(img, vec):
            img_b = np.expand_dims(img, axis=0)
            feed_dict = {x: img_b, w: vec}
            if aux_tensors is not None:
                feed_dict.update(aux_tensors)
            ans = sess.run(jtw, feed_dict=feed_dict)
            return ans
        for i, img in enumerate(img_batch):
            idx = i * hidden_size
            step = hidden_size
            res += grad_func(img, vec[idx:idx + step])
        return res

    def mv(vec):
        tmp_res = []

        def grad_func(img, vec):
            img_b = np.expand_dims(img, axis=0)
            vec = np.squeeze(vec)
            feed_dict = {x: img_b, v: vec, w: np.zeros(hidden_size)}
            if aux_tensors is not None:
                feed_dict.update(aux_tensors)
            ans = sess.run(jv, feed_dict=feed_dict)
            return ans
        for i, img in enumerate(img_batch):
            tmp_res.append(grad_func(img, vec))
        return np.concatenate(tmp_res)
    jac = LinearOperator((hidden_size * batch_size, input_size),
                         matvec=mv, rmatvec=rmv)
    return jac


def power_method(A, p=2, q=2, maxiter=20, verb=0):
    """ Implements power method (alg. 1)

    Args:
        A: scipy.sparse.linalg.LinearOperator
        p: which p-norm to use
        q: which q-norm to use
        maxiter: number of iterations
        verb: logging level

    Returns:
        v: (p, q) - singular vector
        s: the corresponding singular value
    """
    sz = A.shape[1]
    v = np.random.rand(sz) - 0.5
    if p == np.inf:
        v = np.sign(v)
    else:
        v = v / np.linalg.norm(v, ord=p)
        p2 = 1.0 / (1.0 - 1.0 / p)
    for i in range(maxiter):
        Av = A.matvec(v)
        if p == np.inf:
            v = np.sign(A.rmatvec(psi(Av, q)))
        else:
            v = psi(A.rmatvec(psi(Av, q)), p2)
            v = v / np.linalg.norm(v, ord=p)
        if verb > 0:
            s = np.linalg.norm(Av, q)
            print(('Iteration {} / {} with current '
                  'singular value : {}').format(i + 1, maxiter, s))
    s = np.linalg.norm(A.matvec(v), ord=q)
    return v, s
