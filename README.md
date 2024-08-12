# vLLM Lab

An inference environment for launching and benchmarking `vLLM` servers with customizable `yaml` configurations for compatible large language models.

## Docs

- [Main GitHub](https://github.com/vllm-project/vllm)
- [Getting Started](https://docs.vllm.ai/en/latest/index.html)
- [vLLM Serve](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [vLLM Examples](https://docs.vllm.ai/en/latest/getting_started/examples/examples_index.html)

## Quickstart

Install dependencies:

```bash
pip install -r reqirements.txt
```

Copy/paste custom test configurations:

```bash
cp config/vllm_config.yaml config/vllm_test.yaml
```

Launch a vLLM server with your custom test config:

```bash
python server/vllm_server.py config/vllm_test.yaml
```

Test your server API health with the same config file:

 ```bash
python test/vllm_api_health_check.py config/vllm_test.yaml 
```

Run a benchmark (feel free to customize with more options and values):

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

## Default API Usage

| Route | Methods | Description |
| --- | --- | --- |
| `/openapi.json` | GET, HEAD | OpenAPI specification |
| `/docs` | GET, HEAD | Swagger UI for API documentation |
| `/docs/oauth2-redirect` | GET, HEAD | OAuth2 redirect for Swagger UI |
| `/redoc` | GET, HEAD | ReDoc UI for API documentation |
| `/health` | GET | Health check endpoint |
| `/tokenize` | POST | Tokenize input text |
| `/detokenize` | POST | Detokenize token IDs |
| `/v1/models` | GET | List available models |
| `/version` | GET | Get server version information |
| `/v1/chat/completions` | POST | Generate chat completions |
| `/v1/completions` | POST | Generate text completions |

### Benchmark Options

```python
root@a945b872d57c:~/vllm-lab/benchmarks# python3 benchmark_serving.py --help
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

## Server Configuration Options

The `vllm_config.yaml` file wraps all `vllm serve cli` args through a developer friendly config file that can be duplicated and version controlled for varied benchmark settings and model inference performance. [Visit the bottom of this page to learn more about these configuration arguments](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html). Uncomment the configurations you need and adjust the values as desired. Create and save as many config variations as you'd like, editing `server`, `model`, `performance`, and `system` settings to whatever best suits your inference goals.

```yaml
vllm:
  # Server Settings
  server:
    host: 0.0.0.0                            # Bind address for the server
    port: 8000                               # Port number for the vLLM server
    uvicorn_log_level: info                  # Logging level for Uvicorn (debug/info/warning/error/critical/trace)
    # allow_credentials: false                 # Enable CORS credentials
    # allowed_origins: ["*"]                   # CORS allowed origins
    # allowed_methods: ["GET", "POST", "OPTIONS"] # CORS allowed HTTP methods
    # allowed_headers: ["*"]                   # CORS allowed headers
    # api_key: ${VLLM_API_KEY}                 # API key for authentication (from env var)
    # lora_modules: []                         # LoRA module configs (name=path format)
    # prompt_adapters: []                      # Prompt adapter configs (name=path format)
    # chat_template: null                      # Custom chat template file path or inline string
    # response_role: assistant                 # Default role for chat completion responses
    # root_path: null                          # FastAPI root path for proxied setups
    # middleware: []                           # Additional ASGI middleware (import paths)
    # return_tokens_as_token_ids: false        # Return token IDs instead of strings for max_logprobs
    # disable_frontend_multiprocessing: false  # Run OpenAI frontend in same process as model engine

  # Model Configuration
  model:
    name: neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 # HuggingFace model name or local path
    download_dir: /data                      # Directory to download and store model files
    load_format: auto                        # Model loading format (auto/pt/safetensors/npcache/dummy/tensorizer/bitsandbytes)
    dtype: auto                              # Data type for model weights and activations    
    tokenizer: null                          # Custom tokenizer name/path (if different from model)
    tokenizer_mode: auto                     # Tokenizer mode (auto/slow)
    max_model_len: null                      # Override model's maximum context length, should match max_seq_len_to_capture
    kv_cache_dtype: auto                     # Data type for KV cache
    skip_tokenizer_init: false               # Skip tokenizer initialization    
    trust_remote_code: false                 # Trust remote code from HuggingFace   
    # revision: null                           # Specific model version (branch/tag/commit ID)    
    # code_revision: null                      # Specific revision for model code on Hugging Face Hub
    # tokenizer_revision: null                 # Specific tokenizer version

  # Advanced Model Settings
  advanced:
    # quantization: null                       # Method for weight quantization
    # quantization_param_path: null            # Path to JSON file with KV cache scaling factors    
    # enable_lora: false                       # Enable LoRA adapter handling    
    # rope_theta: null                         # RoPE theta parameter
    # tokenizer_pool_size: 0                   # Size of tokenizer pool for async tokenization
    # tokenizer_pool_type: ray                 # Type of tokenizer pool (ray)
    # tokenizer_pool_extra_config: null        # Extra config for tokenizer pool (JSON string)
    # max_loras: 1                             # Maximum number of LoRAs in a single batch
    # max_lora_rank: 16                        # Maximum LoRA rank
    # lora_extra_vocab_size: 256               # Maximum size of extra vocabulary in LoRA adapters
    # lora_dtype: auto                         # Data type for LoRA
    # long_lora_scaling_factors: null          # Scaling factors for multiple LoRA adapters
    # max_cpu_loras: null                      # Maximum number of LoRAs to store in CPU memory
    # fully_sharded_loras: false               # Fully shard LoRA computation
    # enable_prompt_adapter: false             # Enable prompt adapter handling
    # max_prompt_adapters: 1                   # Maximum number of prompt adapters in a batch
    # max_prompt_adapter_token: 0              # Maximum number of prompt adapter tokens
    # scheduler_delay_factor: 0.0              # Delay factor for prompt scheduling
    # guided_decoding_backend: outlines        # Backend for guided decoding (outlines/lm-format-enforcer)
    # num_lookahead_slots: 0                   # Slots for speculative decoding (experimental)
    # max_logprobs: 20                         # Maximum number of log probabilities to return
    # rope_scaling: null                       # RoPE scaling configuration (JSON format)

# Performance Settings (GPU and Compute-related)
  performance:
    device: auto                             # Device type for vLLM execution
    max_seq_len_to_capture: null             # Maximum sequence length for CUDA graph capture    
    gpu_memory_utilization: 0.9              # Fraction of GPU memory to use (0.0 to 1.0)
    max_num_seqs: 256                        # Maximum number of sequences per iteration    
    distributed_executor_backend: mp         # Backend for distributed serving (ray/mp)
    tensor_parallel_size: 1                  # Number of tensor parallel replicas       
    # pipeline_parallel_size: 1                # Number of pipeline parallel stages
    # max_parallel_loading_workers: null       # Number of workers for parallel model loading
    # block_size: 16                           # Token block size for contiguous processing
    # enable_prefix_caching: false             # Enable automatic prefix caching
    # enforce_eager: false                     # Always use eager-mode PyTorch
    # swap_space: 4                            # CPU swap space size (GiB) per GPU
    # cpu_offload_gb: 0                        # Space (GiB) to offload to CPU per GPU
    # disable_sliding_window: false            # Disable sliding window attention
    # use_v2_block_manager: false              # Use BlockSpaceManagerV2
    # num_gpu_blocks_override: null            # Override number of GPU blocks (for testing)
    # max_num_batched_tokens: null             # Maximum number of batched tokens per iteration
    # disable_custom_all_reduce: false         # Disable custom all-reduce implementation
    # enable_chunked_prefill: false            # Enable chunked prefill for large batches
    # speculative_model: null                  # Name of the draft model for speculative decoding
    # num_speculative_tokens: null             # Number of speculative tokens to sample
    # speculative_draft_tensor_parallel_size: null # Tensor parallel replicas for draft model
    # speculative_max_model_len: null          # Maximum sequence length for draft model
    # speculative_disable_by_batch_size: null  # Disable speculative decoding based on batch size
    # ngram_prompt_lookup_max: null            # Max window size for ngram prompt lookup
    # ngram_prompt_lookup_min: null            # Min window size for ngram prompt lookup
    # spec_decoding_acceptance_method: rejection_sampler # Acceptance method for speculative decoding
    # typical_acceptance_sampler_posterior_threshold: null # Posterior threshold for typical acceptance sampler
    # typical_acceptance_sampler_posterior_alpha: null # Alpha for typical acceptance sampler
    # disable_logprobs_during_spec_decoding: null # Disable log probabilities during speculative decoding
    # ray_workers_use_nsight: false            # Use Nsight to profile Ray workers

  # System and Logging Configuration
  system:
    seed: 0                                  # Random config seed for reproducibility
    disable_log_stats: false                 # Disable logging of statistics
    model_loader_extra_config: null          # Extra config for model loader (JSON string)
    ignore_patterns: []                      # Patterns to ignore when loading the model
    preemption_mode: null                    # Preemption mode (recompute/swap)
    served_model_name: []                    # Model name(s) used in the API
    qlora_adapter_name_or_path: null         # Name or path of the QLoRA adapter
    otlp_traces_endpoint: null               # Target URL for OpenTelemetry traces
    engine_use_ray: false                    # Use Ray to start LLM engine in separate process
    disable_log_requests: false              # Disable logging of requests
    max_log_len: null                        # Maximum length for logged prompts

  # Launcher Settings
  launcher:
    log_file: logs/vllm_server.log             # Main log file path
    health_check_directory: logs/health_checks # Directory for health check logs
    log_format: json                           # Log format (json for structured logging)
    log_level: INFO                            # Log level (INFO, DEBUG, etc.)
    log_performance_metrics: true              # Enable logging of performance metrics
    log_rotation_backup_count: 72              # Number of rotated log files to keep
    log_rotation_interval: 1                   # Log rotation interval (hours)
    log_rotation_max_bytes: 104857600          # Maximum log file size before rotation
    performance_log_interval: 30               # Interval for logging performance metrics (seconds)
  
    # Logging Configuration
    logging:
      correlation_id_header: X-Correlation-ID    # Header for request correlation ID
      log_request_details: true                  # Log details of incoming requests
      max_log_len: 1000                          # Maximum length for log messages
      sanitize_log_data: true                    # Remove sensitive data from logs
      health_check:                              # Settings for completions in test/vllm_server_health_check.py
        prompts:
          system_message: "You are an AI Assistant that responds to 'INFERENCE ACKNOWLEDGEMENT REQUESTS:' on route: /v1/chat/completions"
          user_message: "INFERENCE ACKNOWLEDGEMENT REQUEST: Is this endpoint route online and available for requests? Please respond with your acknowledgement in haiku."
```
