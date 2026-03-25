import jax
import jax.numpy as jnp
import beartype
from jaxtyping import Float, jaxtyped

def typecheck(t):
    return jaxtyped(t, typechecker=beartype.beartype)

@typecheck
def test_fn(x: Float[jax.Array, "2 4"]) -> Float[jax.Array, "2 4"]:
    return x

spec = jax.ShapeDtypeStruct((2, 4), jnp.float32)
print("Testing with ShapeDtypeStruct...")
try:
    # This might trigger warnings if jaxtyping or beartype calls jnp.shape on spec
    test_fn(spec)
    print("Success without error")
except Exception as e:
    print(f"Caught exception: {e}")
