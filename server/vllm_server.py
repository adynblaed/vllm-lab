import argparse
import yaml
from typing import Dict, Any, List
import subprocess
import sys
import os
import signal
import logging
import logging.handlers
import json
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class ServerConfig:
    config_path: str
    log_file: str = "logs/vllm_server.log"
    log_level: str = "INFO"
    log_rotation_interval: int = 1  
    log_rotation_backup_count: int = 24 
    log_format: str = "json"
    log_request_details: bool = True
    sanitize_log_data: bool = True
    correlation_id_header: str = "X-Correlation-ID"
    max_log_len: int = 1000

    def __post_init__(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)

        log_file_override = config.get('launcher', {}).get('log_file')
        logs_dir, base_name = os.path.split(log_file_override or self.log_file)
        base_name, ext = os.path.splitext(base_name)

        if not logs_dir:
            logs_dir = 'logs'

        os.makedirs(logs_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y_%m%d_%H%M%S')
        self.log_file = os.path.join(logs_dir, f"{base_name}_{timestamp}{ext}")

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',  
            'name': record.name,
            'level': record.levelname,
            'event': getattr(record, 'event', None), 
        }

        log_message = record.getMessage()
        try:
            parsed_message = json.loads(log_message)
            if isinstance(parsed_message, dict):
                log_data.update(parsed_message)
            else:
                log_data['message'] = log_message
        except json.JSONDecodeError:
            log_data['message'] = log_message

        if record.exc_info:
            log_data['exc_info'] = self.formatException(record.exc_info)

        return json.dumps(log_data)

class VLLMLogger:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("vLLMServer")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))

        file_handler = logging.handlers.TimedRotatingFileHandler(
            self.config.log_file,
            when='H',  
            interval=self.config.log_rotation_interval,
            backupCount=self.config.log_rotation_backup_count
        )
        console_handler = logging.StreamHandler()

        formatter = JsonFormatter() if self.config.log_format == 'json' else logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def log(self, level: str, event: str, **kwargs):
        log_method = getattr(self.logger, level.lower())
        log_data = {'event': event, **kwargs}
        log_method(json.dumps(log_data))

class vLLMServer:
    def __init__(self, server_config: ServerConfig):
        self.server_config = server_config
        self.config = self._load_config()
        self.logger = VLLMLogger(self.server_config)
        self.process = None

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.server_config.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            for section in ['launcher', 'logging']:
                section_config = config.get(section, {})
                for key, value in section_config.items():
                    if hasattr(self.server_config, key):
                        setattr(self.server_config, key, value)
            
            self._replace_env_vars(config)
            return config
        except (yaml.YAMLError, FileNotFoundError) as e:
            self.logger.log('ERROR', 'config_load_error', error=str(e))
            sys.exit(1)

    def _replace_env_vars(self, config: Dict[str, Any]):
        for key, value in config.items():
            if isinstance(value, dict):
                self._replace_env_vars(value)
            elif isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                config[key] = os.getenv(value[2:-1], value)

    def _build_command(self) -> List[str]:
        command = ["vllm", "serve", self.config['vllm']['model']['name']]
        
        supported_args = {
            'host', 'port', 'uvicorn_log_level', 'allow_credentials', 'allowed_origins',
            'allowed_methods', 'allowed_headers', 'api_key', 'lora_modules',
            'prompt_adapters', 'chat_template', 'response_role', 'ssl_keyfile',
            'ssl_certfile', 'ssl_ca_certs', 'ssl_cert_reqs', 'root_path',
            'middleware', 'return_tokens_as_token_ids', 'disable_frontend_multiprocessing',
            'model', 'tokenizer', 'skip_tokenizer_init', 'revision', 'code_revision',
            'tokenizer_revision', 'tokenizer_mode', 'trust_remote_code', 'download_dir',
            'load_format', 'dtype', 'kv_cache_dtype', 'quantization_param_path',
            'max_model_len', 'guided_decoding_backend', 'distributed_executor_backend',
            'worker_use_ray', 'pipeline_parallel_size', 'tensor_parallel_size',
            'max_parallel_loading_workers', 'ray_workers_use_nsight', 'block_size',
            'enable_prefix_caching', 'disable_sliding_window', 'use_v2_block_manager',
            'num_lookahead_slots', 'seed', 'swap_space', 'cpu_offload_gb',
            'gpu_memory_utilization', 'num_gpu_blocks_override', 'max_num_batched_tokens',
            'max_num_seqs', 'max_logprobs', 'disable_log_stats', 'quantization',
            'rope_scaling', 'rope_theta', 'enforce_eager', 'max_context_len_to_capture',
            'max_seq_len_to_capture', 'disable_custom_all_reduce', 'tokenizer_pool_size',
            'tokenizer_pool_type', 'tokenizer_pool_extra_config', 'enable_lora',
            'max_loras', 'max_lora_rank', 'lora_extra_vocab_size', 'lora_dtype',
            'long_lora_scaling_factors', 'max_cpu_loras', 'fully_sharded_loras',
            'enable_prompt_adapter', 'max_prompt_adapters', 'max_prompt_adapter_token',
            'device', 'scheduler_delay_factor', 'enable_chunked_prefill', 'speculative_model',
            'num_speculative_tokens', 'speculative_draft_tensor_parallel_size',
            'speculative_max_model_len', 'speculative_disable_by_batch_size',
            'ngram_prompt_lookup_max', 'ngram_prompt_lookup_min', 'spec_decoding_acceptance_method',
            'typical_acceptance_sampler_posterior_threshold', 'typical_acceptance_sampler_posterior_alpha',
            'disable_logprobs_during_spec_decoding', 'model_loader_extra_config',
            'ignore_patterns', 'preemption_mode', 'served_model_name',
            'qlora_adapter_name_or_path', 'otlp_traces_endpoint', 'engine_use_ray',
            'disable_log_requests', 'max_log_len'
        }
        
        for section, options in self.config['vllm'].items():
            if isinstance(options, dict):
                for key, value in options.items():
                    if key not in supported_args or value is None or (section == 'model' and key == 'name'):
                        continue
                    if key == 'api_key' and not value:
                        continue
                    arg_key = f"--{key.replace('_', '-')}"
                    if isinstance(value, bool):
                        if value:
                            command.append(arg_key)
                    elif isinstance(value, list):
                        command.extend([arg_key, ','.join(map(str, value))])
                    else:
                        command.extend([arg_key, str(value)])
        return command

    def _signal_handler(self, signum, frame):
        self.logger.log('INFO', 'shutdown_initiated', signal=signum)
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.logger.log('WARNING', 'forceful_termination')
                self.process.kill()
        sys.exit(0)

    def run(self):
        command = self._build_command()
        self.logger.log('INFO', 'server_starting', command=' '.join(command))

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            while True:
                output = self.process.stdout.readline()
                if output == '' and self.process.poll() is not None:
                    break
                if output:
                    self.logger.log('INFO', 'server_output', message=output.strip())
            
            for line in self.process.stderr:
                self.logger.log('ERROR', 'server_error', message=line.strip())
            
            return_code = self.process.poll()
            if return_code != 0:
                self.logger.log('ERROR', 'server_exit', return_code=return_code)
                sys.exit(return_code)
        except subprocess.CalledProcessError as e:
            self.logger.log('ERROR', 'server_launch_error', error=str(e))
            sys.exit(1)
        except FileNotFoundError as e:
            self.logger.log('ERROR', 'vllm_not_found', error=str(e))
            sys.exit(1)
        except Exception as e:
            self.logger.log('ERROR', 'unexpected_error', error=str(e))
            sys.exit(1)

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Launch vLLM server with YAML configuration")
    parser.add_argument("config", help="Path to YAML configuration file")
    args = parser.parse_args()
    server_config = ServerConfig(config_path=args.config)
    launcher = vLLMServer(server_config)
    launcher.run()

if __name__ == "__main__":
    main()
