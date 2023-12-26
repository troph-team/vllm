# coding=utf-8

ADAPTER_MAPPING = {
    "LORA": "",
}

MODEL_LAYER_MAPPING ={
    "OPTForCausalLM": "model.decoder.layers",
    "LlamaForCausalLM": "model.layers",
}
