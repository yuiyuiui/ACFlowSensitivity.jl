import jax
import jax.numpy as jnp
from jax import grad

def selu(x,alpha=1.67,lmbda=1.05):
    return lmbda * jnp.where(x>0,x,alpha*(jnp.exp(x)-1))

x=jnp.arange(5.0)
print(selu(x))

def sum_logistic(x):
    return jnp.sum(1.0/(1.0+jnp.exp(-x)))

x_small=jnp.arange(3.)
derivative_logistic=grad(sum_logistic)
print(derivative_logistic(x_small))

def first_finite_difference(f,x,eps=1e-3):
    return jnp.array([(f(x+eps*v)-f(x))/eps for v in jnp.eye(len(x))])

# jnp.eye is used to create a matrix with ones on the diagonal and zeros elsewhere

print(first_finite_difference(sum_logistic,x_small))