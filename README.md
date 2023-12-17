<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
with LoRA support
</h3>

<p align="center">
| <a href="https://vllm.readthedocs.io/en/latest/"><b>Documentation</b></a> | <a href="https://vllm.ai"><b>Blog</b></a> | <a href="https://github.com/vllm-project/vllm/discussions"><b>Discussions</b></a> |

</p>

---

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

llm = LLM(model="meta-llama/Llama-2-7b-hf", load_format="auto", tensor_parallel_size=1, gpu_memory_utilization=0.90)
lora.LoRAModel.from_pretrained(llm.llm_engine.workers[0].model, '/adapter')  # The adapter saved path

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
