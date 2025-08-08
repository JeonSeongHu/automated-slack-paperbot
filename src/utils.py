import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union

import openai
import tqdm
import copy
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

# 최신 OpenAI 클라이언트 초기화
client = openai.OpenAI()

openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    client.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    # chat 계열 모델용 파라미터 이름(신규): max_completion_tokens
    max_completion_tokens: int = 1024
    # text completion 계열 모델용(레거시): max_tokens (없으면 위 값을 사용)
    max_tokens: Optional[int] = None
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    # logprobs: Optional[int] = None


def openai_completion(
    prompts, #: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="text-davinci-003",
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
    ) -> Union[dict, Sequence[dict], Sequence[Sequence[dict]]]:
    """Decode with OpenAI API.

    Args:
        prompts: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        batch_size: Number of prompts to send in a single request. Only for non chat model.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This argument will be deprecated in the future.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - a dict object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    # Treat all modern GPT- chat family (e.g., gpt-3.5/4/5/...) as chat models
    is_chat_model = model_name.startswith("gpt-") or "gpt-3.5" in model_name or "gpt-4" in model_name or "gpt-5" in model_name
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions = []
    for batch_id, prompt_batch in tqdm.tqdm(
        enumerate(prompt_batches),
        desc="prompt_batches",
        total=len(prompt_batches),
    ):
        batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args

        backoff = 3

        while True:
            try:
                # 공통 kwargs (모델별 토큰 파라미터 이름은 아래에서 분기)
                shared_kwargs = dict(
                    model=model_name,
                    temperature=batch_decoding_args.temperature,
                    top_p=batch_decoding_args.top_p,
                    n=batch_decoding_args.n,
                    presence_penalty=batch_decoding_args.presence_penalty,
                    frequency_penalty=batch_decoding_args.frequency_penalty,
                )
                # Optional stop tokens
                if batch_decoding_args.stop:
                    shared_kwargs["stop"] = list(batch_decoding_args.stop)
                if is_chat_model:
                    # chat 모델은 max_completion_tokens 사용
                    # GPT-5는 reasoning 모델이므로 더 많은 토큰이 필요
                    default_tokens = 6144 if "gpt-5" in model_name else 1024
                    tok = batch_decoding_args.max_completion_tokens or default_tokens
                    shared_kwargs["max_completion_tokens"] = tok
                    
                    # logit_bias 추가 (GPT-5는 지원하지 않으므로 제외)
                    if "logit_bias" in decoding_kwargs and "gpt-5" not in model_name:
                        shared_kwargs["logit_bias"] = decoding_kwargs["logit_bias"]
                    

                    system_message = "You are a helpful assistant."
                    
                    # 디버깅: 프롬프트 내용 확인
                    print(f"DEBUG: Model: {model_name}")
                    print(f"DEBUG: Batch ID: {batch_id}")
                    print(f"DEBUG: Prompt batch size: {len(prompt_batch)}")
                    print(f"DEBUG: Max completion tokens: {shared_kwargs.get('max_completion_tokens', 'Not set')}")
                    print(f"DEBUG: User prompt length: {len(prompt_batch[0])}")
                    print(f"DEBUG: Total batches: {len(prompt_batches)}")
                    
                    # GPT-5는 Responses API 사용, 다른 모델은 Chat Completions API 사용
                    if "gpt-5" in model_name:
                        # Responses API 사용
                        response_kwargs = {
                            "model": model_name,
                            "input": prompt_batch[0],
                            "max_output_tokens": shared_kwargs.get("max_completion_tokens", 6144),
                            "reasoning": {"effort": "minimal"},
                            "text": {"verbosity": "low"}
                        }
                        completion_batch = client.responses.create(**response_kwargs)
                    else:
                        # Chat Completions API 사용
                        completion_batch = client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": prompt_batch[0]}
                            ],
                            **shared_kwargs
                        )
                else:
                    # text completion 모델은 max_tokens 사용
                    tok = batch_decoding_args.max_tokens or batch_decoding_args.max_completion_tokens or 1024
                    shared_kwargs["max_tokens"] = tok
                    completion_batch = client.completions.create(prompt=prompt_batch, **shared_kwargs)

                print(f"DEBUG: completion_batch type: {type(completion_batch)}")
                
                # GPT-5 Responses API와 Chat Completions API의 응답 구조 차이 처리
                if "gpt-5" in model_name and hasattr(completion_batch, 'output'):
                    # Responses API 응답 처리
                    print(f"DEBUG: Using Responses API format")
                    print(f"DEBUG: completion_batch.output: {completion_batch.output}")
                    
                    content = None
                    if completion_batch.output and len(completion_batch.output) > 0:
                        # Responses API는 reasoning item과 message item을 포함
                        for output_item in completion_batch.output:
                            print(f"DEBUG: output_item type: {type(output_item)}")
                            print(f"DEBUG: output_item: {output_item}")
                            
                            # ResponseOutputMessage를 찾기
                            if hasattr(output_item, 'type') and output_item.type == 'message':
                                if hasattr(output_item, 'content') and output_item.content:
                                    # content는 리스트이고 첫 번째 항목이 텍스트
                                    if isinstance(output_item.content, list) and len(output_item.content) > 0:
                                        text_item = output_item.content[0]
                                        if hasattr(text_item, 'text'):
                                            content = text_item.text
                                            break
                                    elif hasattr(output_item.content, 'text'):
                                        content = output_item.content.text
                                        break
                    
                    print(f"DEBUG: Final content (Responses): '{content}'")
                    print(f"DEBUG: Content length: {len(content) if content else 0}")
                    
                    # 응답이 끊어졌는지 확인
                    if content and not content.strip().endswith('}'):
                        print("WARNING: Response appears to be truncated!")
                        print(f"WARNING: Last 100 chars: {content[-100:] if len(content) > 100 else content}")
                    
                    # 토큰 사용량 확인
                    if hasattr(completion_batch, "usage") and completion_batch.usage:
                        usage = completion_batch.usage
                        print(f"DEBUG: Token usage - Total: {usage.total_tokens}, Input: {usage.input_tokens if hasattr(usage, 'input_tokens') else 'N/A'}, Output: {usage.output_tokens if hasattr(usage, 'output_tokens') else 'N/A'}")
                        if hasattr(usage, 'output_tokens_details'):
                            details = usage.output_tokens_details
                            print(f"DEBUG: Output details - Reasoning: {details.reasoning_tokens if hasattr(details, 'reasoning_tokens') else 'N/A'}")
                    
                    completions.append({
                        "message": {"content": content},
                        "total_tokens": completion_batch.usage.total_tokens if hasattr(completion_batch, "usage") and completion_batch.usage else None,
                    })
                else:
                    # Chat Completions API 응답 처리
                    print(f"DEBUG: Using Chat Completions API format")
                    print(f"DEBUG: completion_batch.choices length: {len(completion_batch.choices)}")
                    
                    choices = completion_batch.choices
                    for choice in choices:
                        print(f"DEBUG: choice.message.content: {choice.message.content}")
                        print(f"DEBUG: choice.finish_reason: {choice.finish_reason}")
                        
                        content = None
                        if hasattr(choice, "message") and choice.message:
                            content = choice.message.content if hasattr(choice.message, 'content') else None
                        elif hasattr(choice, "text"):
                            content = choice.text
                        
                        print(f"DEBUG: Final content (Chat): '{content}'")
                        
                        completions.append({
                            "message": {"content": content},
                            "total_tokens": completion_batch.usage.total_tokens if hasattr(completion_batch, "usage") and completion_batch.usage else None,
                        })
                break
            except openai.OpenAIError as e:
                logging.warning(f"OpenAIError: {e}.")
                if "Please reduce your prompt" in str(e):
                    if is_chat_model:
                        batch_decoding_args.max_completion_tokens = max(128, int(batch_decoding_args.max_completion_tokens * 0.8))
                        logging.warning(
                            f"Reducing target length to {batch_decoding_args.max_completion_tokens}, Retrying..."
                        )
                    else:
                        cur = batch_decoding_args.max_tokens or batch_decoding_args.max_completion_tokens
                        cur = max(128, int(cur * 0.8))
                        batch_decoding_args.max_tokens = cur
                        logging.warning(
                            f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying..."
                        )
                elif not backoff:
                    logging.error("Hit too many failures, exiting")
                    raise e
                else:
                    backoff -= 1
                    logging.warning("Hit request rate limit; retrying...")
                    time.sleep(sleep_time)  # Annoying rate limit on requests.

    if return_text:
        completions = [c["message"]["content"] for c in completions]
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions


def write_ans_to_file(ans_data, file_prefix, output_dir="./output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, file_prefix + ".txt")
    with open(filename, "w") as f:
        for ans in ans_data:
            f.write(ans + "\n")
