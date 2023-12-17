<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
with LoRA support
</h3>

---

## Environment
**CUDA Version**: 12.2\
**Python Version**: 3.8.10\
**Transformers Version**: 4.34.0\
**bitsandbytes Version**: 0.41.1

```shell
pip install transformers==4.34.0 --upgrade --user
pip install bitsandbytes==0.41.1 --upgrade --user
```

## Compile and install from source

```shell
git clone --branch support_peft https://github.com/SuperBruceJia/vllm.git
cd vllm
pip install -e . --user
```

## Set-up LLM with LoRA 
Please note that this is just a demo!
```python
from vllm import LLM, SamplingParams
from vllm.model_executor.adapters import lora

def stop_token_list():
    stop_tokens = ["Question:",
                   "Question",
                   "USER:",
                   "USER",
                   "ASSISTANT:",
                   "ASSISTANT",
                   "Instruction:",
                   "Instruction",
                   "Response:",
                   "Response",]

    return stop_tokens


stop_tokens = stop_token_list()
sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=128, stop=stop_tokens)

llm = LLM(model="meta-llama/Llama-2-7b-hf", tensor_parallel_size=1, gpu_memory_utilization=0.90)
lora.LoRAModel.from_pretrained(llm.llm_engine.workers[0].model, './adapter')  # The adapter saved path

prompts = ["John writes 20 pages a day.  How long will it take him to write 3 books that are 400 pages each?",
           "Paddington has 40 more goats than Washington. If Washington has 140 goats, how many goats do they have in total?",
           "Ed has 2 dogs, 3 cats and twice as many fish as cats and dogs combined. How many pets does Ed have in total?"]
completions = llm.generate(prompts, sampling_params)
for output in completions:
    gens = output.outputs[0].text
    print(gens, '\n')
```

## Delete LLM and free GPU memory
```python
import gc

import torch
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

# Delete the llm object and free the memory
destroy_model_parallel()
del llm
gc.collect()
torch.cuda.empty_cache()
torch.distributed.destroy_process_group()
print("Successfully delete the llm pipeline and free the GPU memory!")
```

## Download LLaMA 2 models from Meta
```shell
git clone https://github.com/facebookresearch/llama.git
cd llama/
./download.sh
```

## Convert Models to PyTorch `pt` files
```shell
git clone https://github.com/huggingface/transformers.git
cp -r llama/tokenizer.model llama/llama-2-7b
cp -r llama/tokenizer_checklist.chk llama/llama-2-7b
mkdir llama/llama-2-7b-hf
python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir llama/llama-2-7b --model_size 7B --output_dir llama/llama-2-7b-hf
```
![image](https://github.com/SuperBruceJia/vllm/assets/31528604/b8454775-456e-453b-ad9e-e25c4123545c)

