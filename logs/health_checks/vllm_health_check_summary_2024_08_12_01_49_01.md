
# vLLM Server Health Check Summary

```
No LSB modules are available.
Distributor ID:	Ubuntu
Description:	Ubuntu 22.04.4 LTS
Release:	22.04
Codename:	jammy
```


## Endpoint Test Results

**Model:** `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8`

| Endpoint                 | Status                  | Time          |
|--------------------------|-------------------------|---------------|
| /openapi.json      | ✅ OK              | 0.0042s |
| /v1/models         | ✅ OK              | 0.0056s |
| /version           | ✅ OK              | 0.0026s |
| /tokenize          | ✅ OK              | 0.0035s |
| /detokenize        | ✅ OK              | 0.0039s |
| /v1/completions    | ✅ OK              | 0.6105s |
| /v1/chat/completions | ✅ OK              | 0.2860s |
| /metrics           | ✅ OK              | 0.0047s |

**Route Endpoint Lap Time:** `0.9211 seconds`
## Environment Information

- **Kernel Version:** `5.4.0-187-generic`
- **Python Version:** `3.10.12`
- **Torch Version:** `2.4.0+cu121`
- **CUDA Version:** `12.1`
- **vLLM Version:** `0.5.4`

- **Processor:** `x86_64`
- **Memory:** `503.53 GB`
### Consistency Check
*No server configuration inconsistencies detected.*

### Chat Completions Response

>**System Instruction:**
```
You are an AI Assistant that responds to 'INFERENCE ACKNOWLEDGEMENT REQUESTS:' on route: /v1/chat/completions
```

>**Developer Request:**
```
INFERENCE ACKNOWLEDGEMENT REQUEST: Is this endpoint route online and available for requests? Please respond with your acknowledgement in haiku.
```

>**System Response:**
```
Route's online status
Accessible for requests now
Available for use
```

```
Mon Aug 12 01:49:02 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A40                     On  | 00000000:53:00.0 Off |                    0 |
|  0%   46C    P0             195W / 300W |  41306MiB / 46068MiB |     46%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
+---------------------------------------------------------------------------------------+
```

### Log Files

- **Markdown Report:** `logs/health_checks/vllm_health_check_summary_2024_08_12_01_49_01.md`
- **Health Check Log:** `logs/health_checks/vllm_health_check_2024_08_12_01_49_01.log`
- **Sample Metrics:** `logs/health_checks/vllm_metrics_2024_08_12_01_49_01.log`
