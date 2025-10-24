import os
import argparse
import asyncio
import logging
from utils import (
    initialize_servers,
    terminate_servers,
    get_clients,
    generate_research_questions,
    perform_deep_research,
)


async def collect_data(
    vllm_client,
    deeprs_client,
    deeprs_framework="open_deep_research",
    n_question_per_topic=1,
):
    final_reports = []
    for topic in ["liger animal", "tiger habitat", "lion behavior"]:
        for _ in range(n_question_per_topic):
            research_question = await generate_research_questions(
                vllm_client, topic=topic
            )
            final_report = await perform_deep_research(
                deeprs_framework=deeprs_framework,
                deeprs_client=deeprs_client,
                research_question=research_question,
            )
            final_reports.append(final_report)
    return final_reports


def main(args):
    # Initialize Servers
    server_pids = initialize_servers(
        vllm_port=args.vllm_port,
        middleware_port=args.middleware_port,
        deeprs_port=args.deeprs_port,
        deeprs_framework=args.deeprs_framework,
        model_path=args.model_path,
    )

    # Get Clients
    vllm_client, deeprs_client = get_clients(
        vllm_port=args.vllm_port,
        deeprs_port=args.deeprs_port,
        deeprs_framework=args.deeprs_framework,
    )

    # Collect Data
    logging.info("Starting data collection...")
    final_reports = asyncio.run(
        collect_data(
            vllm_client=vllm_client,
            deeprs_client=deeprs_client,
            deeprs_framework=args.deeprs_framework,
            n_question_per_topic=args.n_question_per_topic,
        )
    )

    # Terminate Servers
    terminate_servers(server_pids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model file")
    parser.add_argument("--vllm_port", type=int, help="Port for the vLLM server")
    parser.add_argument(
        "--middleware_port", type=int, help="Port for the middleware server"
    )
    parser.add_argument(
        "--deeprs_port",
        type=int,
        default=2024,
        help="Port for the Deep Researcher server",
    )
    parser.add_argument(
        "--deeprs_framework",
        type=str,
        default="open_deep_research",
        help="Framework for Deep Researcher",
        choices=["open_deep_research"],
    )
    parser.add_argument(
        "--n_question_per_topic",
        type=int,
        default=1,
        help="Number of research questions to generate per topic",
    )
    args = parser.parse_args()
    main(args)
