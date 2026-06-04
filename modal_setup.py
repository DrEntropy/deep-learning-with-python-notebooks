
# used to set up modal notebook. 
# use modal deploy modal_setup.py to deploy the image to modal.
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "keras>=3.13",
        "keras-hub>=0.27",
        "jax",
        "jax[cuda12]",   # GPU JAX on Modal NVIDIA GPUs
        "flax",
        "optax",
        "tensorflow>=2.20",
        "matplotlib",
        "numpy",
    )
    .env({"KERAS_BACKEND": "jax"})
)

app = modal.App(name="chapter15_transformerModal")

@app.function(image=image)
def _image_holder():
    pass