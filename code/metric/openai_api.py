from typing import List
import time
import os
from tqdm import tqdm
import concurrent.futures
import openai
from openai import RateLimitError, APITimeoutError, APIError, APIConnectionError

openai.api_key = os.getenv("OPENAI_API_KEY")

delay_time = 0.5
decay_rate = 0.8


def get_batched_responses(prompts: List[str], model: str, max_tokens: int, batch_size: int, temperature: int = 0,
                          system_message: str = None, histories: List[str] = None, use_parallel: bool = False, show_progress: bool = False):
    responses = []
    it = range(0, len(prompts), batch_size)
    if show_progress:
        it = tqdm(it)
    for batch_start_idx in it:
        batch = prompts[batch_start_idx : batch_start_idx + batch_size]
        histories_batch = histories[batch_start_idx : batch_start_idx + batch_size] if histories else None
        if use_parallel:
            responses += get_parallel_responses(batch, model, max_tokens, temperature=temperature,
                                                system_message=system_message, histories=histories_batch)
        else:
            responses += get_responses(batch, model, max_tokens, temperature=temperature)
    return responses


def get_parallel_responses(prompts: List[str], model: str, max_tokens: int, temperature: int = 0,
                           system_message: str = None, histories: List[dict] = None):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        # Submit requests to threads
        futures = [
            executor.submit(get_responses, [prompt], model, max_tokens, temperature=temperature,
                            system_message=system_message, histories=[histories[prompt_idx]] if histories else None)
            for prompt_idx, prompt in enumerate(prompts)
        ]

        # Wait for all to complete
        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

        # Accumulate results
        results = [future.result()[0] for future in futures]
        return results


def get_responses(prompts: List[str], model="code-davinci-002", max_tokens=400, temperature=0,
                  system_message=None, histories=None, logprobs=None, echo=False):
    global delay_time, cur_key_idx

    # Wait for rate limit
    time.sleep(delay_time)

    # Send request
    try:
        if "gpt-3.5-turbo" in model or "gpt-4" in model:
            results = []
            for prompt_idx, prompt in enumerate(prompts):
                history = histories[prompt_idx] if histories else []
                response = openai.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_message or "You are a helpful assistant."
                        },
                        *history,
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    # request_timeout=45
                )
                results.append(response.choices[0].message.content)
        else:
            response = openai.completions.create(
                model=model,
                prompt=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["\n\n"],
                logprobs=logprobs,
                echo=echo
            )
            results = response["choices"]
        delay_time = max(delay_time * decay_rate, 0.1)
    except (RateLimitError, APITimeoutError, APIError, APIConnectionError) as exc:
        print(openai.api_key, exc)
        delay_time = min(delay_time * 2, 30)
        return get_responses(prompts, model, max_tokens, temperature=temperature, system_message=system_message,
                             histories=histories, logprobs=logprobs, echo=echo)
    except Exception as exc:
        print(exc)
        raise exc

    return results


def get_assistant_responses(prompts: List[str], filenames: List[str], model: str, assistant_name: str, instructions: str):
    # Create assistant
    assistant = openai.beta.assistants.create(
        name=assistant_name,
        instructions=instructions,
        tools=[{"type": "retrieval"}],
        model=model
    )

    results = []
    for prompt, filename in tqdm(zip(prompts, filenames), total=len(prompts)):
        # Attach file
        file = openai.files.create(
            file=open(filename, "rb"),
            purpose="assistants"
        )
        assert len(assistant.file_ids) == 0, "Error: Unknown file attached to assistant"
        assistant = openai.beta.assistants.update(
            assistant_id=assistant.id,
            file_ids=[file.id],
        )
        assert len(assistant.file_ids) == 1, "Error: File not attached"

        # Create thread/run
        thread = openai.beta.threads.create()
        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt,
        )
        run = openai.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )

        # Poll until completed
        while True:
            run = openai.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run.status == "completed":
                break
            time.sleep(2)

        # Get completion
        message_list = openai.beta.threads.messages.list(
            thread_id=thread.id
        )
        results.append(message_list.data[0].content[0].text.value)

        # Delete thread so we don't share any context between prompts
        response = openai.beta.threads.delete(thread.id)
        assert response.deleted, "Error: Thread not deleted"

        # Delete file from assistant
        assistant = openai.beta.assistants.update(
            assistant_id=assistant.id,
            file_ids=[],
        )
        assert len(assistant.file_ids) == 0, "Error: File not deleted from assistant"

    return results