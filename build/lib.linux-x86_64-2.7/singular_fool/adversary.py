from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import singular_fool.utils as utils


def get_adversary(x, y, img_batch, sess, p=np.inf, q=5.0, maxiter=30,
                  adv_norm=10.0, verb=0, aux_tensors=None):
    """Constructs universal adversarial perturbation. Implementation is
    based on https://arxiv.org/pdf/1709.03582.pdf.

    Args:
        x: tf.placeholder specifying the input to the DNN
        y: tf.Tensor, hidden layer of the DNN
        sess: tf.Session
        p: which p-norm to use
        q: which q-norm to use
        maxiter: number of iterations for the Power Method
        adv_norm: desired p-norm of the UAP
        verb: logging level
        aux_tensors: optional, auxilliary tensors to feed into sess (e.g
        test/train phase).
    """

    v, w, jv, jtw = utils.get_gradients(x, y)
    jac = utils.get_jac_matvecs(x, v, w, jv, jtw,
                                sess, img_batch, aux_tensors)
    print("Starting power method...")
    adv, s = utils.power_method(jac, p, q, maxiter, verb=verb)
    return adv_norm * adv / np.linalg.norm(adv, ord=p)
