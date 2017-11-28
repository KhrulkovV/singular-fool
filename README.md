TensorFlow implementation of the algorithm for generating universal adversarial
perturbations based on the (p, q) - singular vectors (see [https://arxiv.org/abs/1709.03582](https://arxiv.org/abs/1709.03582)
for details).

# Installation
For the installation clone the repository and run
```bash
python setup.py install
```
# Usage
An example using Keras for loading pretrained models is given [here](https://github.com/KhrulkovV/singular-fool/blob/master/examples/vgg19.ipynb).

In general, given a DNN with e.g. the following architecture
```python
x = tf.placeholder(tf.float32, (None, 28, 28, 1))
w = tf.Variable(np.random.rand(784, 128), dtype=tf.float32)
fc1 = tf.nn.relu(tf.matmul(tf.reshape(x, (-1, 784)), w))
...
```
it is possible to call
```python
v = singular_fool.get_adversary(x, fc1, img_batch, sess)
```
where
```python
img_batch
```
is a collection of images (e.g. in this case it might be a tensor of size
    [32, 28, 28, 1])
