import argparse
import json
import logging
import os
import re
import requests
import sys
import time
import yaml
from datetime import datetime
from typing import Dict, Any, List, Tuple
from requests.adapters import HTTPAdapter
import psutil
import platform
import torch
import subprocess

class vLLMHealthCheck:
    def __init__(self, config_file: str):
        self.config = self.load_config(config_file)
        self.timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.log_dir = self.config['launcher']['health_check_directory']
        self.log_file = self.setup_logging()
        self.metrics_file = self.setup_metrics_logging()
        self.base_url = f"http://{self.config['vllm']['server']['host']}:{self.config['vllm']['server']['port']}"
        self.model_name = self.config['vllm']['model']['name']
        self.system_message = {"role": "system", "content": self.config['health_check']['prompts']['system_message']}
        self.user_message = {"role": "user", "content": self.config['health_check']['prompts']['user_message']}
        logging.info(f"Loaded system_message: {self.system_message['content']}")
        logging.info(f"Loaded user_message: {self.user_message['content']}")
        self.session = self.create_session()

    @staticmethod
    def load_config(config_file: str) -> Dict[str, Any]:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def setup_logging(self) -> str:
        os.makedirs(self.log_dir, exist_ok=True)
        
        log_file_name = f"vllm_health_check_{self.timestamp}.log"
        log_file = os.path.join(self.log_dir, log_file_name)
        
        logging.basicConfig(
            filename=log_file,
            level=self.config['launcher']['log_level'],
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return log_file

    def get_system_info(self):
        torch_version = torch.__version__
        cuda_version = torch.version.cuda if torch.cuda.is_available() else "Not available"
        
        def run_command(command):
            try:
                output = subprocess.check_output(command, shell=True, universal_newlines=True, stderr=subprocess.STDOUT)
                return output.strip()
            except subprocess.CalledProcessError as e:
                return f"Command failed: {e.output.strip()}"
        
        nvidia_smi = run_command("nvidia-smi")
        nvidia_smi = "\n".join(nvidia_smi.split("\n")[:128])
        
        lsb_release = run_command("lsb_release -a")
        kernel_version = run_command("uname -r")
        
        return torch_version, cuda_version, nvidia_smi, lsb_release, kernel_version

    def setup_metrics_logging(self) -> str:
        metrics_file = os.path.join(self.log_dir, f"vllm_metrics_{self.timestamp}.log")
        return metrics_file

    def create_session(self) -> requests.Session:
        session = requests.Session()
        return session

    def request_with_retries(self, method: str, url: str, retries: int = 3, backoff_factor: float = 0.3, **kwargs) -> requests.Response:
        for attempt in range(retries):
            try:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                logging.warning(f"Request failed (attempt {attempt + 1} of {retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(backoff_factor * (2 ** attempt))
                else:
                    raise
        return None

    def log_response_details(self, endpoint: str, response: requests.Response, elapsed_time: float) -> None:
        logging.info(f"{endpoint}:")
        logging.info(f"Status: {response.status_code}")
        logging.info(f"Elapsed Time: {elapsed_time:.4f} seconds")
        logging.info(f"Headers: {response.headers}")
        logging.info(f"Content: {response.text}")
        logging.info("-" * 40)

    def validate_response_content(self, endpoint: str, content: str) -> Tuple[bool, str]:
        try:
            if endpoint == '/metrics':
                lines = content.strip().split('\n')
                if all(('{' in line and '}' in line and '=' in line.split('=')[1]) or (' ' in line) for line in lines if line and not line.startswith('#')):
                    return True, "Valid metrics content"
                else:
                    return False, "Invalid metrics format"
            else:
                data = json.loads(content)
                if endpoint == '/tokenize':
                    assert 'tokens' in data and isinstance(data['tokens'], list)
                elif endpoint == '/detokenize':
                    assert 'prompt' in data and isinstance(data['prompt'], str)
                elif endpoint == '/v1/models':
                    assert 'data' in data and isinstance(data['data'], list)
                elif endpoint == '/version':
                    assert 'version' in data and isinstance(data['version'], str)
                elif endpoint == '/v1/chat/completions':
                    assert 'choices' in data and isinstance(data['choices'], list)
                return True, "Valid response content"
        except (json.JSONDecodeError, AssertionError) as e:
            logging.error(f"Failed to validate content from {endpoint}. Error: {str(e)}")
            return False, f"Invalid response content: {str(e)}"

    def test_endpoint(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        full_url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        try:
            response = self.request_with_retries(method, full_url, **kwargs)
            elapsed_time = time.time() - start_time
            self.log_response_details(endpoint, response, elapsed_time)
            
            if not response.text.strip():
                logging.warning(f"Empty response from {endpoint}")
                return {
                    "status": response.status_code,
                    "content": "",
                    "elapsed_time": elapsed_time,
                    "is_valid": False,
                    "validation_message": "Empty response"
                }

            is_valid, validation_message = self.validate_response_content(endpoint, response.text)
            return {
                "status": response.status_code,
                "content": response.text,
                "elapsed_time": elapsed_time,
                "is_valid": is_valid,
                "validation_message": validation_message
            }
        except requests.HTTPError as e:
            elapsed_time = time.time() - start_time
            logging.error(f"HTTP Error for {endpoint}: {e.response.status_code} - {e.response.text}")
            return {
                "status": e.response.status_code,
                "content": e.response.text,
                "elapsed_time": elapsed_time,
                "is_valid": False,
                "validation_message": f"HTTP Error: {str(e)}"
            }
        except requests.RequestException as e:
            elapsed_time = time.time() - start_time
            logging.error(f"Error testing {endpoint}: {str(e)}")
            return {
                "status": None,
                "content": str(e),
                "elapsed_time": elapsed_time,
                "is_valid": False,
                "validation_message": f"Request Exception: {str(e)}"
            }

    def parse_metrics(self, metrics_content: str) -> Dict[str, float]:
        metrics = {}
        for line in metrics_content.split('\n'):
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) == 2:
                    metric_name, value = parts
                    try:
                        metrics[metric_name] = float(value)
                    except ValueError:
                        logging.warning(f"Failed to parse metric line: {line}")
        return metrics

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    def run_tests(self) -> Dict[str, Dict[str, Any]]:
        models_response = self.test_endpoint("GET", "/v1/models")
        if models_response['status'] == 200:
            server_model = json.loads(models_response['content'])['data'][0]['id']
            if server_model != self.model_name:
                logging.warning(f"Config model name '{self.model_name}' doesn't match server model '{server_model}'")
        else:
            server_model = self.model_name

        tests = {
            "/openapi.json": lambda: self.test_endpoint("GET", "/openapi.json"),
            "/v1/models": lambda: models_response,
            "/version": lambda: self.test_endpoint("GET", "/version"),
            "/tokenize": lambda: self.test_endpoint("POST", "/tokenize", json={"model": server_model, "prompt": "Hello, world!"}),
            "/detokenize": lambda: self.test_endpoint("POST", "/detokenize", json={"model": server_model, "tokens": [15496, 11, 995, 0]}),
            "/v1/completions": lambda: self.test_endpoint("POST", "/v1/completions", json={
                "model": server_model,
                "prompt": "Tell me a joke only a computer would understand.",
                "max_tokens": 32
            }),            
            "/v1/chat/completions": lambda: self.test_endpoint("POST", "/v1/chat/completions", json={
                "model": server_model,
                "messages": [
                    self.system_message,
                    self.user_message
                ]
            }),
            "/metrics": lambda: self.test_endpoint("GET", "/metrics"),
        }

        results = {}
        for test_name, test_func in tests.items():
            logging.info(f"Running test: {test_name}...")
            result = test_func()
            results[test_name] = result
            status_info = self.get_status_info(result['status'], result['is_valid'])
            logging.info(f"{test_name}: {status_info} (Time: {result['elapsed_time']:.4f}s)")

            if test_name == "/metrics":
                metrics = self.parse_metrics(result['content'])
                self.log_metrics(metrics)

        return results

    @staticmethod
    def get_status_info(status: int, is_valid: bool) -> str:
        if status == 200 and is_valid:
            return "✅ OK"
        elif status == 200 and not is_valid:
            return "⚠️ INVALID RESPONSE"
        elif status is None:
            return "❌ CONNECTION ERROR"
        else:
            return f"❗ ERROR {status}"

    @staticmethod
    def format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
        col_widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]
        header = " | ".join(f"{header:<{width}}" for header, width in zip(headers, col_widths))
        separator = "|".join("-" * width for width in col_widths)
        body = "\n".join(" | ".join(f"{cell:<{width}}" for cell, width in zip(row, col_widths)) for row in rows)
        return f"{header}\n{separator}\n{body}"

    def parse_server_metrics(self) -> Dict[str, Any]:
        with open(self.log_file, 'r') as f:
            log_content = f.read()
        
        metrics = {}
        metric_patterns = {
            'prompt_throughput': r'Avg prompt throughput: (\d+\.\d+) tokens/s',
            'generation_throughput': r'Avg generation throughput: (\d+\.\d+) tokens/s',
            'gpu_usage': r'GPU KV cache usage: (\d+\.\d+)%',
            'cpu_usage': r'CPU KV cache usage: (\d+\.\d+)%'
        }
        for metric, pattern in metric_patterns.items():
            match = re.search(pattern, log_content)
            if match:
                metrics[metric] = float(match.group(1))
        return metrics

    @staticmethod
    def model_names_match(name1: str, name2: str) -> bool:
        return name1.lower().replace('-', '').replace('_', '') == name2.lower().replace('-', '').replace('_', '')

    def check_consistency(self, results: Dict[str, Dict[str, Any]]) -> List[str]:
        inconsistencies = []
        model_id = json.loads(results['/v1/models']['content'])['data'][0]['id']
        
        for test in ['/tokenize', '/detokenize', '/v1/chat/completions']:
            test_model = json.loads(results[test]['content']).get('model')
            if test_model and not self.model_names_match(test_model, model_id):
                inconsistencies.append(f"Model mismatch in {test}: Expected {model_id}, got {test_model}")

        if not self.model_names_match(self.model_name, model_id):
            inconsistencies.append(f"Model mismatch: Config specifies '{self.model_name}', but server reports '{model_id}'")
        
        return inconsistencies

    def print_summary(self, results: Dict[str, Dict[str, Any]], report_path: str) -> None:
        torch_version, cuda_version, nvidia_smi, lsb_release, kernel_version = self.get_system_info()

        summary = []

        summary.append("# vLLM Server Health Check Summary\n")
        summary.append(f"```\n{lsb_release}\n```\n")

        summary.append("\n## Endpoint Test Results")
        model_info = f"\n**Model:** `{json.loads(results['/v1/models']['content'])['data'][0]['id']}`\n"
        summary.append(model_info)
        
        summary.append("| Endpoint                 | Status                  | Time          |")
        summary.append("|--------------------------|-------------------------|---------------|")

        table_data = []
        total_time = 0
        added_tests = set()

        for test_name, data in results.items():
            if test_name in added_tests:
                continue

            status = data['status']
            elapsed_time = data['elapsed_time']
            total_time += elapsed_time

            status_info = self.get_status_info(status, data['is_valid'])
            
            table_data.append(f"| {test_name:<18} | {status_info:<17} | {elapsed_time:.4f}s |")
            added_tests.add(test_name)

        summary.extend(table_data)
        
        summary.append(f"\n**Route Endpoint Lap Time:** `{total_time:.4f} seconds`") 

        summary.append("## Environment Information\n")
        summary.append(f"- **Kernel Version:** `{kernel_version}`")
        summary.append(f"- **Python Version:** `{platform.python_version()}`")
        summary.append(f"- **Torch Version:** `{torch_version}`")
        summary.append(f"- **CUDA Version:** `{cuda_version}`")
        version_info = f"- **vLLM Version:** `{json.loads(results['/version']['content'])['version']}`"
        summary.append(f"{version_info}\n")
        summary.append(f"- **Processor:** `{platform.processor()}`")
        summary.append(f"- **Memory:** `{psutil.virtual_memory().total / (1024 ** 3):.2f} GB`")
                        
        summary.append("### Consistency Check")
        inconsistencies = self.check_consistency(results)
        if inconsistencies:
            summary.append("**Server configuration inconsistencies detected:**\n")
            for inconsistency in inconsistencies:
                summary.append(f"- {inconsistency}")
        else:
            summary.append("*No server configuration inconsistencies detected.*\n")
            
        summary.append("### Chat Completions Response\n")
        chat_completion_result = results.get("/v1/chat/completions")
        if chat_completion_result:
            response_content = json.loads(chat_completion_result['content'])

            # Use dynamic variables for system and user messages
            system_message_content = self.system_message['content']
            user_message_content = self.user_message['content']
            assistant_response = response_content['choices'][0]['message']['content']

            summary.append(f">**System Instruction:**\n```\n{system_message_content}\n```\n")
            summary.append(f">**Developer Request:**\n```\n{user_message_content}\n```\n")
            summary.append(f">**System Response:**\n```\n{assistant_response}\n```\n")

            summary.append(f"```\n{nvidia_smi}\n```\n")

            summary.append("### Log Files\n")
            summary.append(f"- **Markdown Report:** `{report_path}`")
            summary.append(f"- **Health Check Log:** `{self.log_file}`")
            summary.append(f"- **Sample Metrics:** `{self.metrics_file}`")

        console_and_markdown_output = "\n".join([
            "",         
            *summary,    
            "",          
        ])

        print(console_and_markdown_output)

        self.export_markdown(console_and_markdown_output, report_path)

    def export_markdown(self, content: str, report_path: str):

        with open(report_path, 'w') as f:
            f.write(content)

    def run(self):
        try:
            results = self.run_tests()

            report_path = os.path.join(self.log_dir, f"vllm_health_check_summary_{self.timestamp}.md")
            
            self.print_summary(results, report_path)
        except Exception as e:
            logging.error(f"Error during health check: {str(e)}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Test vLLM server endpoints and display a comprehensive summary.")
    parser.add_argument("config_file", help="Path to the YAML configuration file")
    args = parser.parse_args()

    health_check = vLLMHealthCheck(args.config_file)
    health_check.run()

if __name__ == "__main__":
    main()
    