from scripts.convert_lora_weights import map_hf_key_to_mlx, map_mlx_key_to_hf

def test_key_mapping():
    hf_key = "base_model.model.model.layers.5.self_attn.q_proj.lora_A.weight"
    mlx_key = map_hf_key_to_mlx(hf_key)
    assert mlx_key == "model.layers.5.self_attn.q_proj.lora_a"

    back_to_hf = map_mlx_key_to_hf(mlx_key)
    assert back_to_hf == hf_key
