# Benchmarking vLLM

## Downloading a test ShareGPT dataset

You can download a test dataset by running:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Run a benchmark (feel free to insert custom options and values):

```python
python3 benchmarks/benchmark_serving.py \
    --backend openai \
    --base-url http://0.0.0.0:8000 \
    --model neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
    --num-prompts 1000 \
    --dataset-name sharegpt  \
    --dataset-path benchmarks/data/ShareGPT_V3_cleaned_split_test_dataset.json \
    --request-rate 1000 \
    --seed 12345
```

### Benchmark Options

```python
root@a945b872d57c:~/vllm-benchmark-arena/benchmarks# python3 benchmark_serving.py --help
usage: benchmark_serving.py [-h]
                            [--backend {tgi,vllm,lmdeploy,deepspeed-mii,openai,openai-chat,tensorrt-llm,scalellm}]
                            [--base-url BASE_URL] [--host HOST] [--port PORT] [--endpoint ENDPOINT]
                            [--dataset DATASET] [--dataset-name {sharegpt,sonnet,random}]
                            [--dataset-path DATASET_PATH] --model MODEL [--tokenizer TOKENIZER]
                            [--best-of BEST_OF] [--use-beam-search] [--num-prompts NUM_PROMPTS]
                            [--sharegpt-output-len SHAREGPT_OUTPUT_LEN] [--sonnet-input-len SONNET_INPUT_LEN]
                            [--sonnet-output-len SONNET_OUTPUT_LEN] [--sonnet-prefix-len SONNET_PREFIX_LEN]
                            [--random-input-len RANDOM_INPUT_LEN] [--random-output-len RANDOM_OUTPUT_LEN]
                            [--random-range-ratio RANDOM_RANGE_RATIO] [--request-rate REQUEST_RATE]
                            [--seed SEED] [--trust-remote-code] [--disable-tqdm] [--save-result]
                            [--metadata [KEY=VALUE ...]] [--result-dir RESULT_DIR]
                            [--result-filename RESULT_FILENAME]

Benchmark the online serving throughput.

options:
  -h, --help            show this help message and exit
  --backend {tgi,vllm,lmdeploy,deepspeed-mii,openai,openai-chat,tensorrt-llm,scalellm}
  --base-url BASE_URL   Server or API base url if not using http host and port.
  --host HOST
  --port PORT
  --endpoint ENDPOINT   API endpoint.
  --dataset DATASET     Path to the ShareGPT dataset, will be deprecated in the next release.
  --dataset-name {sharegpt,sonnet,random}
                        Name of the dataset to benchmark on.
  --dataset-path DATASET_PATH
                        Path to the dataset.
  --model MODEL         Name of the model.
  --tokenizer TOKENIZER
                        Name or path of the tokenizer, if not using the default tokenizer.
  --best-of BEST_OF     Generates `best_of` sequences per prompt and returns the best one.
  --use-beam-search
  --num-prompts NUM_PROMPTS
                        Number of prompts to process.
  --sharegpt-output-len SHAREGPT_OUTPUT_LEN
                        Output length for each request. Overrides the output length from the ShareGPT
                        dataset.
  --sonnet-input-len SONNET_INPUT_LEN
                        Number of input tokens per request, used only for sonnet dataset.
  --sonnet-output-len SONNET_OUTPUT_LEN
                        Number of output tokens per request, used only for sonnet dataset.
  --sonnet-prefix-len SONNET_PREFIX_LEN
                        Number of prefix tokens per request, used only for sonnet dataset.
  --random-input-len RANDOM_INPUT_LEN
                        Number of input tokens per request, used only for random sampling.
  --random-output-len RANDOM_OUTPUT_LEN
                        Number of output tokens per request, used only for random sampling.
  --random-range-ratio RANDOM_RANGE_RATIO
                        Range of sampled ratio of input/output length, used only for random sampling.
  --request-rate REQUEST_RATE
                        Number of requests per second. If this is inf, then all the requests are sent at time
                        0. Otherwise, we use Poisson process to synthesize the request arrival times.
  --seed SEED
  --trust-remote-code   Trust remote code from huggingface
  --disable-tqdm        Specify to disable tqdm progress bar.
  --save-result         Specify to save benchmark results to a json file
  --metadata [KEY=VALUE ...]
                        Key-value pairs (e.g, --metadata version=0.3.3 tp=1) for metadata of this run to be
                        saved in the result JSON file for record keeping purposes.
  --result-dir RESULT_DIR
                        Specify directory to save benchmark json results.If not specified, results are saved
                        in the current directory.
  --result-filename RESULT_FILENAME
                        Specify the filename to save benchmark json results.If not specified, results will be
                        saved in {backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json format.
```

For more details, check out [`api_server.py` in the vLLM source repo](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py) and the [`benchmarks` from the latest main branch](https://github.com/vllm-project/vllm/tree/main/benchmarks). If you run into weird problems or compatibility hurdles, [visit vLLM's `issues` page](https://github.com/vllm-project/vllm/issues) to see if there's a similar shared error others have reported.
