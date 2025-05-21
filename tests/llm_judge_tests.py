import json
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


class JudgeContinuous(BaseModel):
    score: float
    feedback: str


class JudgeDiscrete(BaseModel):
    any_items_meet_criteria: bool
    feedback: str


evals = [
    ("clarity", JudgeContinuous),
    ("sensationalism", JudgeDiscrete),
    ("engagement", JudgeContinuous),
    ("relevance", JudgeContinuous),
]


def llm_judge_evals(
    output_path: str, eval_name: str, eval_output_format: Union[JudgeContinuous, JudgeDiscrete]
) -> Union[JudgeContinuous, JudgeDiscrete]:
    llm = OpenAI()

    with open(output_path, "r") as f:
        data = json.load(f)

    with open(f"prompts/evals/{eval_name}.txt", "r") as f:
        prompt = f.read()

    if eval_name == "relevance":
        llm_input = f"""PROFILE: {output_path.split("_output")[0]}
CANDIDATE POOL: {data["candidate_pool"]}
RECOMMENDED HEADLINES: {"\n".join([str(item["rank"]) + ". " + item["headline"] for item in data["recommendations"]])}
"""
    else:
        llm_input = "\n".join(f"{item['rank']}. {item['headline']}" for item in data["recommendations"])

    response = llm.responses.parse(
        model="gpt-4.1", instructions=prompt, input=llm_input, temperature=0, text_format=eval_output_format
    )

    return response.output_parsed


def diversity_eval(output_path: str) -> float:
    llm = OpenAI()
    with open(output_path, "r") as f:
        data = json.load(f)

    headlines = [item["headline"] for item in data["recommendations"]]
    embeddings = []

    for h in headlines:
        response = llm.embeddings.create(input=h, model="text-embedding-3-small")
        embeddings.append(response.data[0].embedding)

    # Calculate cosine similarity between all pairs of embeddings
    # and compute the average similarity score
    similarity_matrix = cosine_similarity(embeddings)
    avg_similarity = similarity_matrix.mean()
    # Invert the score to get a diversity score
    diversity_score = 1 - avg_similarity

    return diversity_score


def rewrite_delta_eval(output_path: str) -> float:
    llm = OpenAI()
    with open(output_path, "r") as f:
        data = json.load(f)

    headlines_original = [item["original_headline"] for item in data["recommendations"]]
    headlines_rewritten = [item["headline"] for item in data["recommendations"]]
    # Get the embeddings for both original and rewritten headlines
    embeddings_original = []
    embeddings_rewritten = []
    for h in headlines_original:
        response = llm.embeddings.create(input=h, model="text-embedding-3-small")
        embeddings_original.append(response.data[0].embedding)
    for h in headlines_rewritten:
        response = llm.embeddings.create(input=h, model="text-embedding-3-small")
        embeddings_rewritten.append(response.data[0].embedding)
    # Calculate cosine similarity between corresponding pairs of embeddings
    # and compute the average similarity score
    similarity_matrix = cosine_similarity(embeddings_original, embeddings_rewritten)
    avg_similarity = similarity_matrix.mean()

    return avg_similarity


def run_evals(output_path: str):
    eval_results = []

    diversity_eval_score = diversity_eval(output_path)
    rewrite_delta_eval_score = rewrite_delta_eval(output_path)

    eval_results.append({"diversity": diversity_eval_score})
    eval_results.append({"rewrite_delta": rewrite_delta_eval_score})

    for eval_name, eval_output_format in evals:
        eval_output = llm_judge_evals(output_path, eval_name, eval_output_format)

        eval_results.append({eval_name: eval_output.model_dump_json()})
    return eval_results


if __name__ == "__main__":
    output_paths = Path("data").glob("*_output.json")
    for output_path in output_paths:
        print(f"Running evaluations for {output_path}")
        eval_results = run_evals(str(output_path))
        with open(f"data/{output_path.stem}_evals.json", "w") as output_file:
            json.dump(eval_results, output_file, indent=2)
