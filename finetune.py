"""Instruction fine-tuning of Gemma3 1B with LoRA on the Databricks Dolly 15k dataset.

Extracted from `chapter16_text-generation.ipynb` (Deep Learning with Python, 3rd ed.),
"Instruction fine-tuning" section. Runs Keras 3 on the JAX backend, intended for a
single A100-80GB GPU (e.g. a RunPod pod with a persistent /workspace volume).

Install dependencies (TF cpu build is fine; it is only used for tf.data / tokenizer):
    pip install -U "jax[cuda12]" keras keras-hub tensorflow-cpu kagglehub python-dotenv

Kaggle credentials (Gemma is gated) are read from a .env file next to this script,
e.g.:
    KAGGLE_USERNAME=your_username
    KAGGLE_KEY=your_key

Run detached so a dropped SSH connection does not kill training:
    nohup python finetune.py &
"""

import os

# Load credentials (e.g. KAGGLE_USERNAME / KAGGLE_KEY) from a .env file, if present.
from dotenv import load_dotenv

load_dotenv()

# Must be set before importing keras / tensorflow.
os.environ["KERAS_BACKEND"] = "jax"
# Free up more GPU memory on the JAX and TensorFlow backends.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import json

import keras
import keras_hub
import tensorflow as tf

import kagglehub

# Persistent volume on the pod. Everything we want to keep is written here.
WORKSPACE = "/workspace"


def main():
    # Gemma is gated on Kaggle. For a headless run, put KAGGLE_USERNAME and
    # KAGGLE_KEY in the .env file (loaded above) or the environment; these are
    # picked up automatically. Otherwise fall back to interactive login.
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        kagglehub.login()

    # --- Load the pretrained model -----------------------------------------
    gemma_lm = keras_hub.models.CausalLM.from_preset(
        "gemma3_1b",
        dtype="float32",
    )

    # --- Build the instruction-tuning dataset ------------------------------
    PROMPT_TEMPLATE = """"[instruction]\n{}[end]\n[response]\n"""
    RESPONSE_TEMPLATE = """{}[end]"""

    dataset_path = keras.utils.get_file(
        origin=(
            "https://hf.co/datasets/databricks/databricks-dolly-15k/"
            "resolve/main/databricks-dolly-15k.jsonl"
        ),
    )
    data = {"prompts": [], "responses": []}
    with open(dataset_path) as file:
        for line in file:
            features = json.loads(line)
            if features["context"]:
                continue
            data["prompts"].append(PROMPT_TEMPLATE.format(features["instruction"]))
            data["responses"].append(RESPONSE_TEMPLATE.format(features["response"]))

    ds = tf.data.Dataset.from_tensor_slices(data).shuffle(2000).batch(2)
    val_ds = ds.take(100)
    train_ds = ds.skip(100)

    preprocessor = gemma_lm.preprocessor
    preprocessor.sequence_length = 512

    # --- Low-Rank Adaptation (LoRA) ----------------------------------------
    gemma_lm.backbone.enable_lora(rank=8)

    # --- Compile and fit ----------------------------------------------------
    gemma_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(5e-5),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(WORKSPACE, "finetuned_best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.CSVLogger(os.path.join(WORKSPACE, "training_log.csv")),
    ]

    gemma_lm.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        callbacks=callbacks,
    )

    # --- Persist the fine-tuned model --------------------------------------
    gemma_lm.save(os.path.join(WORKSPACE, "finetuned.keras"))
    gemma_lm.save_weights(os.path.join(WORKSPACE, "finetuned.weights.h5"))
    print("Saved fine-tuned model to", WORKSPACE)


if __name__ == "__main__":
    main()
