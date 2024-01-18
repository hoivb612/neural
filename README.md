# Neural Speed

Neural Speed is an innovation library designed to provide the efficient inference of large language models (LLMs) on Intel platforms through the state-of-the-art (SOTA) low-bit quantization and sparsity powered by [Intel Neural Compressor](https://github.com/intel/neural-compressor) and [llama.cpp](https://github.com/ggerganov/llama.cpp). We provide the experimental features as below:

- Modular design to support new models
- [Highly optimized low precision kernels](neural_speed/core/README.md)
- Utilize AMX, VNNI, AVX512F and AVX2 instruction set
- Support CPU (x86 platforms only) and Intel GPU (WIP)
- Support 4bits and 8bits quantization

> Neural Speed is under active development so APIs are subject to change.

You can refer [this blog](https://medium.com/@NeuralCompressor/llm-performance-of-intel-extension-for-transformers-f7d061556176) to get the performance information.

## Quick Start
There are two approaches for utilizing the Neural Speed:
### 1. Transformer-like API

NeuralSpeed
```
from transformers import AutoTokenizer, TextStreamer
from neural_speed import Model

prompt = "Once upon a time, a little girl"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = Model()
model.init(model_name, weight_dtype="int4", compute_dtype="int8", group_size=32)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=30, do_sample=True)
```

You can also use [Transformer-based API](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/weightonlyquant.md#llm-runtime-example-code) in [ITREX(intel extension for transformers)](https://github.com/intel/intel-extension-for-transformers), but you need to install Intel Extension for Transformers.

### 2. Neural Speed straight forward:

#### One-click Python scripts
Run LLM with one-click python script including conversion, quantization and inference.
```
python scripts/run.py model-path --weight_dtype int4 -p "She opened the door and see"
```

#### Quantize and Inference Step By Step
Besides the one-click script, Neural Speed also offers the detailed script: 1) convert and quantize, and 2) inference.

##### 1. Convert and Quantize LLM
converting the model by following the below steps:

```bash
# convert the model directly use model id in Hugging Face. (recommended)
python scripts/convert.py --outtype f32 --outfile ne-f32.bin EleutherAI/gpt-j-6b
```

Neural Speed assumes the compatible model format as [Q4_0, Q5_0, Q8_0 GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md). You can directly use GGUF models from HuggingFace instead of converting. Validated GGUF models from HuggnigFace: [llama2-7b](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF), [Mistral-7B](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF), [CodeLlama-7B](https://huggingface.co/TheBloke/CodeLlama-7B-GGUF), [CodeLlama-13B](https://huggingface.co/TheBloke/CodeLlama-13B-GGUF). 

Neural Speed also supports GGUF models generated by [llama.cpp](https://github.com/ggerganov/llama.cpp), you need to download the model and use llama.cpp to create it. Validated models: [llama2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [falcon-7b](https://huggingface.co/tiiuae/falcon-7b), [falcon-40b](https://huggingface.co/tiiuae/falcon-40b), [mpt-7b](https://huggingface.co/mosaicml/mpt-7b), [mpt-40b](https://huggingface.co/mosaicml/mpt-40b) and [bloom-7b1](https://huggingface.co/bigscience/bloomz-7b1).

##### 2. Inference

We provide LLM inference script to run the quantized model. Please reach [us](mailto:itrex.maintainers@intel.com) if you want to run using C++ API directly.
```bash
# recommed to use numactl to bind cores in Intel cpus for better performance
# if you use different core numbers, please also  change -t arg value
# please type prompt about codes when run `StarCoder`, for example, -p "def fibonnaci(".

#Linux and WSL
OMP_NUM_THREADS=<physic_cores> numactl -m 0 -C 0-<physic_cores-1> python scripts/inference.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t <physic_cores> --color -p "She opened the door and see"

# if you want to generate fixed outputs, please set --seed arg, for example:
OMP_NUM_THREADS=<physic_cores> numactl -m 0 -C 0-<physic_cores-1> python scripts/inference.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t <physic_cores> --color -p "She opened the door and see" --seed 12

# if you want to reduce repeated generated texts, please set --repeat_penalty (value > 1.0, default = 1.0), for example:
OMP_NUM_THREADS=<physic_cores> numactl -m 0 -C 0-<physic_cores-1> python scripts/inference.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t <physic_cores> --color -p "She opened the door and see" --repeat_penalty 1.2

#Windows
#Recommend to build and run our project in WSL to get a better and stable performance
python scripts/inference.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t <physic_cores|P-cores> --color -p "She opened the door and see"
```

> For details please refer to [Advanced Usage](./docs/advanced_usage.md).

## Supported Hardware
| Hardware | Optimization |
|-------------|:-------------:|
|Intel Xeon Scalable Processors | ✔ |
|Intel Xeon CPU Max Series | ✔ |
|Intel Core Processors | ✔ |
|Intel Arc GPU Series | WIP |
|Intel Data Center GPU Max Series | WIP |
|Intel Gaudi2 | Not yet |

## Supported Models

Neural Speed supports the following models: [list](./docs/supported_models.md)

## Install

### Build Python package
```shell
pip install .
```

### Build executable only

```shell
# Linux and WSL
git submodule update --init --recursive
mkdir build
cd build
cmake .. -G Ninja
ninja
```

```powershell
# Windows
# Install VisualStudio 2022 and open 'Developer PowerShell for VS 2022'
mkdir build
cd build
cmake ..
cmake --build . -j --config Release
```


## Advanced Usage

### 1. Quantization and inferenece
You can do quantization and inference without Transformer-API: [Advanced Usage](./docs/advanced_usage.md).

### 2. Tensor Parallelism cross nodes/sockets

We support tensor parallelism strategy for distributed inference/training on multi-node and multi-socket. You can refer to [tensor_parallelism.md](./docs/tensor_parallelism.md) to enable this feature.


### 3. Contribution

You can consider adding your own models, please follow the document: [graph developer document](./developer_document.md). 

### 4. Custom Stopping Criteria

You can customize the stopping criteria according to your own needs by processing the input_ids to determine if text generation needs to be stopped.
Here is the document of Custom Stopping Criteria: [simple example with minimum generation length of 80 tokens](./docs/customized_stop.md)

### 5. Verbose Mode

Enable verbose mode and control tracing information using the `NEURAL_SPEED_VERBOSE` environment variable.

Available modes:
- 0: Print all tracing information. Comprehensive output, including: evaluation time and operator profiling.
- 1: Print evaluation time. Time taken for each evaluation.
- 2: Profile individual operator. Identify performance bottleneck within the model.
