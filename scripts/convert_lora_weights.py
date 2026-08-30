import re
import os
import json

def map_hf_key_to_mlx(key: str) -> str:
    k = key.replace("base_model.model.", "")
    k = k.replace(".lora_A.weight", ".lora_a")
    k = k.replace(".lora_B.weight", ".lora_b")
    return k

def map_mlx_key_to_hf(key: str) -> str:
    k = "base_model.model." + key
    k = k.replace(".lora_a", ".lora_A.weight")
    k = k.replace(".lora_b", ".lora_B.weight")
    return k

def convert_hf_to_mlx(hf_dir: str, mlx_path: str):
    print(f"Converting HF PEFT adapter at {hf_dir} to MLX at {mlx_path}")

def convert_mlx_to_hf(mlx_path: str, hf_dir: str):
    print(f"Converting MLX adapter at {mlx_path} to HF PEFT at {hf_dir}")

def main():
    print("Bidirectional LoRA Converter module initialized.")

if __name__ == "__main__":
    main()
