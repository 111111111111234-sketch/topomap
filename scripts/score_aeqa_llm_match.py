#!/usr/bin/env python3
"""Score A-EQA predictions with a GPT-4o judge through Requesty's OpenAI API."""

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = """You are a strict evaluator for embodied question answering.
For every item, compare the candidate answer against the reference answer(s), using
only semantic correctness. Ignore wording, capitalization, and harmless extra
detail. A contradiction to the reference is incorrect. Return 100 only for a
semantically correct answer, 0 for an incorrect answer, and an integer 1-99 only
when the candidate is genuinely partially correct. `correct` must be true only
when the score is 100. Return JSON only in this exact shape:
{"results":[{"question_id":"...","score":0,"correct":false,"rationale":"brief"}]}.
Every input question_id must appear exactly once."""


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=root / "data" / "aeqa_questions-41.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=root / "results" / "exp_eval_aeqa" / "gpt_answer_merged_unique.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "exp_eval_aeqa" / "aeqa_llm_match.json",
    )
    parser.add_argument("--model", default="openai/gpt-4o")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def reference_answers(item):
    return [item["answer"], *item.get("extra_answers", [])]


def load_existing(path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {item["question_id"]: item for item in data.get("items", [])}


def save(path, questions, items, metadata):
    ordered = [items[q["question_id"]] for q in questions if q["question_id"] in items]
    complete = len(ordered) == len(questions)
    scores = [item["score"] for item in ordered]
    correct = [item["correct"] for item in ordered]
    answered = [item for item in ordered if item["candidate_answer"] is not None]
    payload = {
        "metadata": metadata,
        "total_questions": len(ordered),
        "expected_total_questions": len(questions),
        "complete": complete,
        "answered_questions": len(answered),
        "llm_match_all_questions": (
            sum(scores) / len(scores) if complete else None
        ),
        "llm_match_answered_only": (
            sum(item["score"] for item in answered) / len(answered) if answered else 0
        ),
        "binary_accuracy_all_questions": (
            sum(correct) / len(correct) if complete else None
        ),
        "binary_accuracy_answered_only": (
            sum(item["correct"] for item in answered) / len(answered)
            if answered
            else 0
        ),
        "items": ordered,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def grade_batch(client, model, batch):
    user_prompt = json.dumps({"items": batch}, ensure_ascii=False)
    last_error = None
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            results = data["results"]
            expected = {item["question_id"] for item in batch}
            actual = {item["question_id"] for item in results}
            if actual != expected or len(results) != len(batch):
                raise ValueError("grader returned incomplete or duplicate question IDs")
            return results
        except Exception as exc:  # Keep retries explicit for transient Requesty errors.
            last_error = exc
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"grading batch failed after 5 attempts: {last_error}")


def main():
    args = parse_args()
    api_key = os.environ.get("REQUESTY_API_KEY")
    if not api_key:
        raise SystemExit("REQUESTY_API_KEY is not set")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    questions = json.loads(args.questions.read_text())
    predictions = {
        item["question_id"]: item["answer"]
        for item in json.loads(args.predictions.read_text())
    }
    items = load_existing(args.output)
    metadata = {
        "judge_model": args.model,
        "judge_endpoint": os.environ.get(
            "REQUESTY_BASE_URL", "https://router.requesty.ai/v1"
        ),
        "questions": str(args.questions),
        "predictions": str(args.predictions),
        "failed_episodes_score": 0,
        "scoring_note": "Independent GPT-4o judge; the paper's exact grader prompt is not released.",
    }

    # Evaluation failures have no candidate answer and are deterministically zero.
    pending = []
    for question in questions:
        qid = question["question_id"]
        if qid in items:
            continue
        candidate = predictions.get(qid)
        if candidate is None:
            items[qid] = {
                "question_id": qid,
                "question": question["question"],
                "reference_answers": reference_answers(question),
                "candidate_answer": None,
                "score": 0,
                "correct": False,
                "rationale": "No answer: episode failed under the fixed evaluation configuration.",
            }
        else:
            pending.append(
                {
                    "question_id": qid,
                    "question": question["question"],
                    "reference_answers": reference_answers(question),
                    "candidate_answer": candidate,
                }
            )

    client = OpenAI(
        base_url=os.environ.get("REQUESTY_BASE_URL", "https://router.requesty.ai/v1"),
        api_key=api_key,
    )
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        for graded in grade_batch(client, args.model, batch):
            source = next(item for item in batch if item["question_id"] == graded["question_id"])
            score = int(graded["score"])
            if not 0 <= score <= 100:
                raise ValueError(f"invalid score for {source['question_id']}: {score}")
            items[source["question_id"]] = {
                **source,
                "score": score,
                "correct": bool(graded["correct"]) and score == 100,
                "rationale": str(graded.get("rationale", "")),
            }
        save(args.output, questions, items, metadata)
        print(f"graded {min(start + len(batch), len(pending))}/{len(pending)} answered items")

    save(args.output, questions, items, metadata)
    report = json.loads(args.output.read_text())
    print(f"LLM-Match (all 41): {report['llm_match_all_questions']:.2f}")
    print(f"Binary accuracy (all 41): {report['binary_accuracy_all_questions']:.2%}")


if __name__ == "__main__":
    main()
