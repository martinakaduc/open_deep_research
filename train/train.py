from typing import List

import os
import argparse
import asyncio
import logging
import json
from generate_ideas import generate_next_idea, check_idea_novelty
from utils import (
    initialize_servers,
    terminate_servers,
    get_clients,
    get_model_configs,
    generate_research_questions,
    perform_deep_research,
)

logging.basicConfig(level=logging.INFO)


async def collect_data(
    round_idx: int,
    topics: List[str],
    vllm_client,
    model_name: str,
    deeprs_client,
    deeprs_framework: str = "open_deep_research",
    n_questions: int = 1,
    num_reflections: int = 3,
    skip_novelty_check: bool = False,
    paper_search_engine: str = "semanticscholar",
):
    base_dir = "data/topics/{topic}"
    result_dir = f"data/round_{round_idx}"
    n_questions_per_topic = n_questions // len(topics)

    ideas = []
    for topic in topics:
        topic_ideas = []
        topic_base_dir = base_dir.format(topic=topic)
        # Load previous ideas if not the first round
        n_prev_ideas = 0
        if round_idx > 0:
            pre_result_dir = f"data/round_{round_idx - 1}"
            prev_idea_archive = []
            with open(os.path.join(pre_result_dir, "ideas.json"), "r") as f:
                seed_ideas = json.load(f)
            for seed_idea in seed_ideas:
                prev_idea_archive.append(json.dumps(seed_idea))

            topic_ideas.extend(prev_idea_archive)
            n_prev_ideas = len(prev_idea_archive)

        for _ in range(n_questions_per_topic):
            topic_ideas = await generate_next_idea(
                base_dir=topic_base_dir,
                result_dir=result_dir,
                client=vllm_client,
                model=model_name,
                prev_idea_archive=topic_ideas,
                num_reflections=num_reflections,
            )

        if not skip_novelty_check:
            topic_ideas = await check_idea_novelty(
                ideas=topic_ideas,
                base_dir=base_dir,
                result_dir=result_dir,
                client=vllm_client,
                model=model_name,
                engine=paper_search_engine,
            )

        # Only keep new ideas in this round
        ideas.extend(topic_ideas[n_prev_ideas:])

    final_reports = []
    for idea in ideas:
        research_question = await generate_research_questions(
            vllm_client, topic=idea, model_name=model_name
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
    model_configs = get_model_configs(args.model_path)

    # Collect Data
    logging.info("Starting data collection...")
    for round_idx in range(args.n_rounds):
        logging.info(f"Starting round {round_idx + 1}/{args.n_rounds}...")
        final_reports = asyncio.run(
            collect_data(
                round_idx=round_idx,
                topics=args.topics,
                vllm_client=vllm_client,
                model_name=model_configs["model_name"],
                deeprs_client=deeprs_client,
                deeprs_framework=args.deeprs_framework,
                n_questions=args.n_questions,
                num_reflections=args.num_reflections,
                skip_novelty_check=args.skip_novelty_check,
                paper_search_engine=args.paper_search_engine,
            )
        )

    # Terminate Servers
    terminate_servers(server_pids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model file")
    parser.add_argument(
        "--vllm_port",
        type=int,
        default=8081,
        help="Port for the vLLM server",
    )
    parser.add_argument(
        "--middleware_port",
        type=int,
        default=8082,
        help="Port for the middleware server",
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
        "--n_rounds",
        type=int,
        default=1,
        help="Number of improvement rounds",
    )
    parser.add_argument(
        "--topics",
        type=str,
        nargs="+",
        choices=["seir", "earthquake-prediction"],
        help="List of research topics to start with",
    )
    parser.add_argument(
        "--n_questions",
        type=int,
        default=5,
        help="Number of research questions to generate per round",
    )
    parser.add_argument(
        "--num_reflections",
        type=int,
        default=1,
        help="Number of reflections for question generation",
    )
    parser.add_argument(
        "--skip_novelty_check",
        action="store_true",
        help="Whether to skip novelty check for generated questions",
    )
    parser.add_argument(
        "--paper_search_engine",
        type=str,
        default="semanticscholar",
        help="Paper search engine to use for novelty check",
        choices=["semanticscholar", "openalex"],
    )
    args = parser.parse_args()
    main(args)
