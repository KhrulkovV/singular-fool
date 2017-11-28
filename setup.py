from setuptools import setup

setup(name='singular_fool',
      version='0.1.0',
      description=('universal adversarial perturbations based on'
                   '(p, q)-singular vectors'),
      url='https://github.com/KhrulkovV/singular_fool',
      author='Valentin Khrulkov',
      author_email='khrulkov.v@gmail.com',
      license='MIT',
      packages=['singular_fool'],
      install_requires=['numpy', 'scipy'],
      zip_safe=False)
