"""
run:
python -m relevancy run_all_day_paper \
  --output_dir ./data \
  --model_name="gpt-3.5-turbo-16k" \
"""
import time
import json
import os
import random
import re
import string
import yaml
from datetime import datetime
from structured_output import coerce_and_validate

import numpy as np
import tqdm
import utils


def load_prompt_config(config_path="config.yaml"):
    """Load prompt configuration from yaml file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("prompt_config", {})


def build_domain_hint(prompt_config):
    """Build domain-specific hint from config."""
    researcher_profile = prompt_config.get("researcher_profile", {})
    
    specialization = researcher_profile.get("specialization", "")
    high_priority_topics = researcher_profile.get("high_priority_topics", [])
    deprioritize_note = researcher_profile.get("deprioritize_note", "")
    sota_bonus_note = researcher_profile.get("sota_bonus_note", "")
    
    output_format = prompt_config.get("output_format", {})
    json_schema = output_format.get("json_schema", "")
    scoring_guidance = output_format.get("scoring_guidance", "")
    
    domain_hint = f"\nYou are selecting papers for a researcher specializing in {specialization}.\n"
    domain_hint += "HIGH PRIORITY topics include:\n"
    
    for topic in high_priority_topics:
        domain_hint += f"• {topic}\n"
    
    if deprioritize_note:
        domain_hint += f"{deprioritize_note}\n"
    
    if sota_bonus_note:
        domain_hint += f"{sota_bonus_note}\n"
    
    domain_hint += "For each paper, output one JSON object per line with fields: \n"
    domain_hint += f"{json_schema}\n"
    
    if scoring_guidance:
        domain_hint += f"Scoring guidance: {scoring_guidance}\n"
    
    return domain_hint


def build_venue_hint(prompt_config):
    """Build venue-specific hint from config."""
    venue_config = prompt_config.get("venue_config", {})
    top_tier_venues = venue_config.get("top_tier_venues", [])
    venue_bonus_note = venue_config.get("venue_bonus_note", "")
    
    venue_list = ", ".join(top_tier_venues)
    venue_hint = f"\nIf the 'comments' mention a top-tier venue (e.g., {venue_list})\n"
    venue_hint += f"{venue_bonus_note}\n"
    
    return venue_hint


def encode_prompt(query, prompt_papers, config_path="config.yaml"):
    """Encode multiple prompt instructions into a single string."""
    # Load prompt configuration from config file
    prompt_config = load_prompt_config(config_path)
    
    # Load base prompt template
    base_prompt_file = prompt_config.get("base_prompt_file", "relevancy_prompt.txt")
    base = open(f"src/{base_prompt_file}").read()
    
    # Build domain and venue hints from config
    domain_hint = build_domain_hint(prompt_config)
    venue_hint = build_venue_hint(prompt_config)
    
    # Add output format instructions
    output_format = prompt_config.get("output_format", {})
    output_instructions = output_format.get("output_instructions", "")
    language_note = output_format.get("language_note", "")
    
    prompt = base + "\n" + domain_hint + venue_hint
    if output_instructions:
        prompt += f"\n{output_instructions}\n"
    if language_note:
        prompt += f"{language_note}\n"
    
    prompt += "\n" + (query['interest'] or "")

    for idx, task_dict in enumerate(prompt_papers):
        (title, authors, abstract) = task_dict["title"], task_dict["authors"], task_dict["abstract"]
        if not title:
            raise
        prompt += f"###\n"
        prompt += f"{idx + 1}. Title: {title}\n"
        prompt += f"{idx + 1}. Authors: {authors}\n"
        prompt += f"{idx + 1}. Abstract: {abstract}\n"
    prompt += f"\n Generate response:\n1."
    print(prompt)
    return prompt


def post_process_chat_gpt_response(paper_data, response, threshold_score=8):
    selected_data = []
    if response is None:
        return []
    content = response['message']['content']
    print(f"DEBUG: Processing content length: {len(content)}")
    print(f"DEBUG: Content ends with: '{content[-100:] if len(content) > 100 else content}'")
    
    # Strip code fences
    content = content.replace("```json", "").replace("```", "")
    score_items = []
    # First try: extract all JSON objects anywhere in the text
    candidates = re.findall(r"\{[\s\S]*?\}", content)
    print(f"DEBUG: Found {len(candidates)} JSON candidates")
    for obj in candidates:
        try:
            item = json.loads(obj)
            if isinstance(item, dict) and any(k.lower() == "relevancy score" for k in item.keys()):
                ok, fixed, _ = coerce_and_validate(item)
                if ok:
                    score_items.append(fixed)
        except Exception:
            pass
    # Second try: line-wise extracting substring between first '{' and last '}'
    if not score_items:
        for line in content.splitlines():
            line = line.strip()
            if "{" not in line:
                continue
            try:
                start = line.find("{")
                end = line.rfind("}")
                if end > start:
                    obj = line[start:end+1]
                    item = json.loads(obj)
                    if isinstance(item, dict) and any(k.lower() == "relevancy score" for k in item.keys()):
                        ok, fixed, _ = coerce_and_validate(item)
                        if ok:
                            score_items.append(fixed)
            except Exception:
                continue
    # Fallback: nothing parsed; return empty selection with no hallucination
    if not score_items:
        return [], False
    # Normalize scores
    scores = []
    for item in score_items:
        temp = item["Relevancy score"]
        if isinstance(temp, str) and "/" in temp:
            scores.append(int(temp.split("/")[0]))
        else:
            scores.append(int(temp))
    # Align lengths
    hallucination = len(score_items) != len(paper_data)
    max_len = min(len(score_items), len(paper_data))
    for idx in range(max_len):
        print(f"DEBUG: Paper {idx+1} score: {scores[idx]}, threshold: {threshold_score}")
        if scores[idx] < threshold_score:
            print(f"DEBUG: Skipping paper {idx+1} (score {scores[idx]} < threshold {threshold_score})")
            continue
        print(f"DEBUG: Including paper {idx+1} (score {scores[idx]} >= threshold {threshold_score})")
        inst = score_items[idx]
        output_str = "Title: " + paper_data[idx]["title"] + "\n"
        output_str += "Authors: " + paper_data[idx]["authors"] + "\n"
        output_str += "Link: " + paper_data[idx]["main_page"] + "\n"
        for key, value in inst.items():
            paper_data[idx][key] = value
            output_str += str(key) + ": " + str(value) + "\n"
        paper_data[idx]['summarized_text'] = output_str
        selected_data.append(paper_data[idx])
    return selected_data, hallucination


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def process_subject_fields(subjects):
    # Normalize subjects like "Subjects:\nComputation and Language (cs.CL); Artificial Intelligence (cs.AI)"
    normalized = subjects.replace("Subjects:\n", "").replace("Subjects:", "")
    parts = [p.strip() for p in normalized.split(";") if p.strip()]
    clean = [p.split(" (")[0].strip() for p in parts]
    return clean

def generate_relevance_score(
    all_papers,
    query,
    model_name="gpt-5-mini",
    threshold_score=8,
    num_paper_in_prompt=3,
    top_p=1.0,
    sorting=True,
    config_path="config.yaml"
):
    ans_data = []
    request_idx = 1
    hallucination = False
    for id in tqdm.tqdm(range(0, len(all_papers), num_paper_in_prompt)):
        prompt_papers = all_papers[id:id+num_paper_in_prompt]
        # only sampling from the seed tasks
        # Append per-paper hints (venue/project) from comments when available
        enriched = []
        for p in prompt_papers:
            note = ""
            c = p.get("comments", "")
            if c:
                note += f"\n[meta] comments: {c}"
            proj = p.get("project_url", "")
            if proj:
                note += f"\n[meta] project: {proj}"
            enriched.append({"title": p["title"], "authors": p["authors"], "abstract": p["abstract"] + note})
        prompt = encode_prompt(query, enriched, config_path)

        decoding_args = utils.OpenAIDecodingArguments(
            temperature=1.0,
            n=1,
            max_completion_tokens=6144 if "gpt-5" in model_name else 1024,
            top_p=top_p,
        )
        request_start = time.time()
        response = utils.openai_completion(
            prompts=prompt,
            model_name=model_name,
            batch_size=1,
            decoding_args=decoding_args,
            logit_bias={"100257": -100},  # prevent the <|endoftext|> from being generated
        )
        raw_content = response['message']['content']
        print ("response", raw_content)
        print(f"DEBUG: Raw content length: {len(raw_content) if raw_content else 0}")
        print(f"DEBUG: Raw content ends with: '{raw_content[-50:] if raw_content and len(raw_content) > 50 else raw_content}'")
        
        if os.getenv("DIGEST_DEBUG", "0") == "1":
            try:
                os.makedirs("debug", exist_ok=True)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                with open(f"debug/llm_response_{ts}_{id}.txt", "w") as df:
                    df.write(raw_content)
                print(f"DEBUG: Wrote {len(raw_content)} characters to debug file")
            except Exception as e:
                print(f"DEBUG: Error writing debug file: {e}")
                pass
        request_duration = time.time() - request_start

        process_start = time.time()
        batch_data, hallu = post_process_chat_gpt_response(prompt_papers, response, threshold_score=threshold_score)
        hallucination = hallucination or hallu
        ans_data.extend(batch_data)

        print(f"Request {request_idx+1} took {request_duration:.2f}s")
        print(f"Post-processing took {time.time() - process_start:.2f}s")

    if sorting:
        ans_data = sorted(ans_data, key=lambda x: int(x["Relevancy score"]), reverse=True)
    
    return ans_data, hallucination

def run_all_day_paper(
    query,
    date=None,
    data_dir="../data",
    model_name="gpt-5-mini",
    threshold_score=8,
    num_paper_in_prompt=3,
    temperature=0.4,
    top_p=1.0
):
    if date is None:
        date = datetime.today().strftime('%a, %d %b %y')
        # string format such as Wed, 10 May 23
    print ("the date for the arxiv data is: ", date)

    all_papers = [json.loads(l) for l in open(f"{data_dir}/{date}.jsonl", "r")]
    print (f"We found {len(all_papers)}.")

    all_papers_in_subjects = [
        t for t in all_papers
        if bool(set(process_subject_fields(t['subjects'])) & set(query['subjects']))
    ]
    print(f"After filtering subjects, we have {len(all_papers_in_subjects)} papers left.")
    ans_data, hallucination = generate_relevance_score(all_papers_in_subjects, query, model_name, threshold_score, num_paper_in_prompt, top_p)
    utils.write_ans_to_file(ans_data, date, output_dir="../outputs")
    return ans_data


if __name__ == "__main__":
    query = {"interest":"""
    1. Computer Vision and 3D Vision: 3D reconstruction, multi-view geometry, NeRF, Gaussian Splatting, novel view synthesis, SLAM, point clouds
    2. Vision-Language Models (VLMs): multimodal learning, vision-text understanding, visual reasoning, CLIP-like models
    3. Novel Architectures: new neural network designs, attention mechanisms, transformers for vision, efficient architectures
    4. Representation Learning: self-supervised learning, contrastive learning, foundation models, pre-training strategies
    5. Diffusion Models: score-based generative modeling for images/3D, training/inference efficiency\n""",
    "subjects":["Computer Vision and Pattern Recognition", "Artificial Intelligence", "Machine Learning"]}
    ans_data = run_all_day_paper(query)
