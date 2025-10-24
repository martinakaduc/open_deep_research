import os
import time
import psutil
import requests
import subprocess
import logging
from typing import Any, Dict


def run_server(cmd_string):
    try:
        server_process = subprocess.Popen(cmd_string, shell=True)
        return server_process
    except Exception as e:
        print(f"Error starting server: {e}")
        return None


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def shutdown_server(process):
    try:
        kill(process.pid)
        # process.terminate()
        print("Server shutdown successfully.")
    except Exception as e:
        print(f"Error shutting down server: {e}")


def get_model_configs(model_path: str):
    if model_path == "Qwen/Qwen2.5-Omni-7B":
        return {
            "model_name": "vllm:qwen-2.5-omni-7b",
            "max_model_len": 32768,
            "tensor_parallel_size": 1,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
        }
    else:
        raise ValueError(f"Unsupported model path: {model_path}")


def check_health(url):
    server_ok = False
    while server_ok is False:
        try:
            # Send a GET request to the health check endpoint
            response = requests.get(url)

            # Check if the server is healthy
            if response.status_code == 200:
                server_ok = True
            else:
                time.sleep(1)

        except requests.exceptions.RequestException as e:
            time.sleep(1)
    return server_ok


def initialize_servers(
    model_path: str,
    vllm_port: int,
    middleware_port: int,
    deeprs_port: int,
    deeprs_framework: str = "open_deep_research",
    middleware_workers: int = 2,
) -> Dict[str, Any]:
    model_configs = get_model_configs(model_path)

    # Start vLLM Server
    logging.info("Starting vLLM server...")
    vllm_pid = run_server(
        (
            f"vllm serve {model_path} "
            f"--served-model-name {model_configs['model_name']} "
            f"--port {vllm_port} "
            f"--max-model-len {model_configs['max_model_len']} "
            f"--tensor-parallel-size {model_configs['tensor_parallel_size']} "
            f"--enable-auto-tool-choice {model_configs['enable_auto_tool_choice']} "
            f"--tool-call-parser {model_configs['tool_call_parser']} &"
        )
    )

    check_health(f"http://localhost:{vllm_port}/health")

    # Start Middleware Server
    logging.info("Starting Middleware server...")
    middleware_pid = run_server(
        (
            f"python wrapper_server.py "
            f"--port {middleware_port} "
            f"--vllm-url http://localhost:{vllm_port}/v1 "
            f"--workers {middleware_workers} &"
        )
    )

    time.sleep(5)  # Give middleware time to start

    # Start Deep Researcher Server
    logging.info("Starting Deep Researcher server...")
    if deeprs_framework == "open_deep_research":
        deeprs_pid = run_server(
            (
                f'SUMMARIZATION_MODEL="{model_configs['model_name']}" '
                f'RESEARCH_MODEL="{model_configs['model_name']}" '
                f'COMPRESSION_MODEL="{model_configs['model_name']}" '
                f'FINAL_REPORT_MODEL="{model_configs['model_name']}" '
                f'SUMMARIZATION_MODEL_BASE_URL="http://localhost:{middleware_port}/v1" '
                f'SUMMARIZATION_MODEL_PROVIDER="openai" '
                f'RESEARCH_MODEL_BASE_URL="http://localhost:{middleware_port}/v1" '
                f'RESEARCH_MODEL_PROVIDER="openai" '
                f'COMPRESSION_MODEL_BASE_URL="http://localhost:{middleware_port}/v1" '
                f'COMPRESSION_MODEL_PROVIDER="openai" '
                f'FINAL_REPORT_MODEL_BASE_URL="http://localhost:{middleware_port}/v1" '
                f'FINAL_REPORT_MODEL_PROVIDER="openai" '
                f'uvx --refresh --from "langgraph-cli[inmem]" --with-editable ../../ --python 3.11 langgraph dev --allow-blocking &'
            )
        )
    else:
        raise ValueError(f"Unsupported Deep Researcher framework: {deeprs_framework}")

    time.sleep(5)  # Give Deep Researcher time to start

    return {
        "vllm_pid": vllm_pid,
        "middleware_pid": middleware_pid,
        "deeprs_pid": deeprs_pid,
    }


def terminate_servers(pids: Dict[str, Any]):
    shutdown_server(pids["middleware_pid"])
    shutdown_server(pids["vllm_pid"])
    shutdown_server(pids["deeprs_pid"])


def get_clients(
    vllm_port: int, deeprs_port: int, deeprs_framework: str = "open_deep_research"
):
    from openai import OpenAI

    openai_client = OpenAI(
        base_url=f"http://localhost:{vllm_port}/v1",
    )

    if deeprs_framework == "open_deep_research":
        from langgraph_sdk import get_client

        deeprs_client = get_client(url=f"http://localhost:{deeprs_port}")
    else:
        raise ValueError(f"Unsupported Deep Researcher framework: {deeprs_framework}")

    return openai_client, deeprs_client


async def generate_research_questions(client, topic: str):
    output = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {
                "role": "user",
                "content": f"Generate a research question about {topic}. Return only the question and nothing else.",
            }
        ],
        temperature=1.0,
    )
    return output.choices[0].message.content


async def perform_deep_research(
    deeprs_framework,
    deeprs_client,
    research_question: str,
):
    if deeprs_framework == "open_deep_research":
        async for chunk in deeprs_client.runs.stream(
            None,  # Threadless run
            "Deep Researcher",  # Name of assistant. Defined in langgraph.json.
            input={
                "messages": [
                    {
                        "role": "human",
                        "content": research_question,
                    }
                ],
            },
            stream_mode="updates",
        ):
            json_data = chunk.data
            if "final_report_generation" in json_data:
                final_report = json_data["final_report_generation"]["report"]
                print("Final Report:", final_report)
                return final_report
    else:
        raise ValueError(f"Unsupported Deep Researcher framework: {deeprs_framework}")
