# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Language Model that uses Together AI api.

Recommended model names are:
  'google/gemma-2-9b-it'
  'google/gemma-2-27b-it'
"""

from collections.abc import Collection, Sequence
import concurrent.futures
import os
import random
import time

from absl import logging
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
import numpy as np
import together
from typing_extensions import override

_MAX_ATTEMPTS = 20
_NUM_SILENT_ATTEMPTS = 3
_SECONDS_TO_SLEEP_WHEN_RATE_LIMITED = 2
_JITTER_SECONDS = 0.25
_DEFAULT_NUM_RESPONSE_TOKENS = 5000

_GUESS_CHARS_PER_TOKEN = 4
# Max tokens is really 8193, but we leave substantial margin since estimates
# of the number of tokens are imprecise and also calculated before adding the
# system messages.
_MAX_ALLOWED_TOKENS = 7000
# _MAX_ALLOWED_TOKENS = 64000
# Use `_NUM_INITIAL_TOKENS` from the start of the prompt if possible when
# trimming to fit the whole sequence into `_MAX_ALLOWED_TOKENS`.
_NUM_INITIAL_TOKENS = 500


def _ensure_prompt_not_too_long(
    prompt: str,
    num_response_tokens: int,
    guess_chars_per_token: int = _GUESS_CHARS_PER_TOKEN,
) -> str:
    r"""Ensures the prompt is not too long for Together AI\'s Gemma-2 models."""
    num_initial_chars = _NUM_INITIAL_TOKENS * guess_chars_per_token
    max_prompt_tokens = _MAX_ALLOWED_TOKENS - num_response_tokens
    if max_prompt_tokens <= 0:
        raise ValueError(
            f"Cannot reserve {num_response_tokens} of {_MAX_ALLOWED_TOKENS} tokens."
        )
    max_prompt_chars = max_prompt_tokens * guess_chars_per_token
    if len(prompt) <= max_prompt_chars:
        return prompt

    # Keep the first _NUM_INITIAL_TOKENS tokens and then skip to the last tokens
    # and take as many as we can from the end.
    if max_prompt_chars > num_initial_chars:
        num_final_chars = max_prompt_chars - num_initial_chars
        new_prompt = prompt[:num_initial_chars] + prompt[-num_final_chars:]
        logging.info(
            "Prompt too long, trimmed it down, while keeping start and "
            "end, resulting in %d characters",
            len(new_prompt),
        )
        logging.debug("Trimmed prompt: %s", new_prompt)
        return new_prompt

    # This happens if len(prompt) > max_prompt_chars <= num_initial_chars.
    new_prompt = prompt[-max_prompt_chars:]
    logging.info(
        "Prompt too long, truncated it to last %d characters.", max_prompt_chars
    )
    logging.debug("Truncated prompt: %s", new_prompt)
    return new_prompt


class Gemma2(language_model.LanguageModel):
    """Language Model that uses Together AI models."""

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        measurements: measurements_lib.Measurements | None = None,
        channel: str = language_model.DEFAULT_STATS_CHANNEL,
    ):
        """Initializes the instance.

        Args:
          model_name: The language model to use. For more details, see
            https://api.together.xyz/models.
          api_key: The API key to use when accessing the Together AI API. If None,
            will use the TOGETHER_AI_API_KEY environment variable.
          measurements: The measurements object to log usage statistics to.
          channel: The channel to write the statistics to.
        """
        if api_key is None:
            api_key = os.environ["TOGETHER_AI_API_KEY"]
        self.base_url = os.environ.get("TOGETHER_AI_BASE_URL", None)
        self._api_key = api_key
        self._model_name = model_name
        self._measurements = measurements
        self._channel = channel
        self.errors = []
        if not self.base_url:
          self._client = together.Together(
              api_key=self._api_key,
              base_url=self.base_url,
          )
          self.errors = [
              together.error.APIError,
              together.error.RateLimitError,
              together.error.ServiceUnavailableError,
          ]
        else:
            import openai
            self._client = openai.OpenAI(
                api_key=self._api_key,
                base_url=self.base_url,
            )
            self.errors = [
                openai.APIError,
                openai.RateLimitError,
            ]

    @override
    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
        temperature: float = language_model.DEFAULT_TEMPERATURE,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        original_prompt = prompt
        prompt = _ensure_prompt_not_too_long(prompt, max_tokens)
        messages = [
            {
                "role": "system" if not self.base_url else "user",
                "content": (
                    "You always continue sentences provided "
                    "by the user and you never repeat what "
                    "the user has already said. All responses must end with a "
                    "period. Try not to use lists, but if you must, then "
                    "always delimit list items using either "
                    r"semicolons or single newline characters ('\n'), never "
                    r"delimit list items with double carriage returns ('\n\n')."
                    + "\n\nQuestion: Is Jake a turtle?\nAnswer: Jake is "
                    if self.base_url
                    else ""
                    # "\n\nQuestion: Is Jake a turtle?\nAnswer: Jake is "
                ),
            },
        ]
        if not self.base_url:
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": "Question: Is Jake a turtle?\nAnswer: Jake is ",
                    },
                ]
            )

        messages.extend(
            [
                {"role": "assistant", "content": "not a turtle."},
                {
                    "role": "user",
                    "content": (
                        "Question: What is Priya doing right now?\nAnswer: "
                        + "Priya is currently "
                    ),
                },
                {"role": "assistant", "content": "sleeping."},
                {"role": "user", "content": prompt},
            ]
        )

        # print(f"messages: {messages}")

        # gemma2 does not support `tokens` + `max_new_tokens` > 8193.
        # gemma2 interprets our `max_tokens`` as their `max_new_tokens`.
        max_tokens = min(max_tokens, _DEFAULT_NUM_RESPONSE_TOKENS)

        result = ""
        for attempts in range(_MAX_ATTEMPTS):
            if attempts > 0:
                seconds_to_sleep = _SECONDS_TO_SLEEP_WHEN_RATE_LIMITED + random.uniform(
                    -_JITTER_SECONDS, _JITTER_SECONDS
                )
                if attempts >= _NUM_SILENT_ATTEMPTS:
                    print(
                        f"Sleeping for {seconds_to_sleep} seconds... "
                        + f"attempt: {attempts} / {_MAX_ATTEMPTS}"
                    )
                time.sleep(seconds_to_sleep)
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    stop=terminators,
                    seed=seed,
                    stream=False,
                )
            except tuple(self.errors) as err:
                if attempts >= _NUM_SILENT_ATTEMPTS:
                    print(f"  Exception: {err}")
                    print(f"  Text exception prompt: {prompt}")
                if isinstance(err, self.errors[0]):
                    # If hit the error that arises from a prompt that is too long then
                    # re-run the trimming function with a more pessimistic guess of the
                    # the number of characters per token.
                    prompt = _ensure_prompt_not_too_long(
                        original_prompt, max_tokens, guess_chars_per_token=1
                    )
                continue
            else:
                result = response.choices[0].message.content
                break

        if self._measurements is not None:
            self._measurements.publish_datum(
                self._channel,
                {"raw_text_length": len(result)},
            )

        return result

    @override
    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, dict[str, float]]:
        augmented_prompt = _ensure_prompt_not_too_long(prompt, 1)
        messages = [
            {
                "role": "system" if not self.base_url else "user",
                "content": (
                    "You always continue sentences provided "
                    "by the user and you never repeat what "
                    "the user already said."
                    + "\n\nQuestion: Is Jake a turtle?\nAnswer: Jake is "
                    if self.base_url
                    else ""
                ),
            }
        ]
        if not self.base_url:
            messages.append({
                "role": "user",
                "content": "Question: Is Jake a turtle?\nAnswer: Jake is ",
            })
        messages.extend([
            {"role": "assistant", "content": "not a turtle."},
            {
                "role": "user",
                "content": "Question: What is Priya doing right now?\nAnswer: Priya is currently ",
            },
            {"role": "assistant", "content": "sleeping."},
            {"role": "user", "content": augmented_prompt},
        ])

        result = None
        for attempts in range(_MAX_ATTEMPTS):
            if attempts > 0:
                seconds_to_sleep = _SECONDS_TO_SLEEP_WHEN_RATE_LIMITED + random.uniform(-_JITTER_SECONDS, _JITTER_SECONDS)
                if attempts >= _NUM_SILENT_ATTEMPTS:
                    print(f"Sleeping for {seconds_to_sleep} seconds.. attempt: {attempts} / {_MAX_ATTEMPTS}")
                time.sleep(seconds_to_sleep)
            try:
                lp_args = {
                    "logprobs": True,
                    "top_logprobs": 20,
                } if self.base_url else {"logprobs": 1}
                result = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    max_tokens=1,
                    seed=seed,
                    stream=False,
                    **lp_args,
                )
                break
            except tuple(self.errors) as err:
                if attempts >= _NUM_SILENT_ATTEMPTS:
                    print(f"  Exception: {err}")
                    print(f"  Choice exception prompt: {augmented_prompt}")
                continue

        if not result:
            raise ValueError(f"Failed to get logprobs.\nException prompt: {augmented_prompt}")

        logprobs = {}
        if self.base_url:
            top_logprobs = {
                token.token: token.logprob
                for token in result.choices[0].logprobs.content
            }
        else:
            top_logprobs = {
                token: logprob
                for token, logprob in zip(result.choices[0].logprobs.tokens, result.choices[0].logprobs.token_logprobs)
            }

        parsed_responses = []
        for response in responses:
            matching_logprob = None
            # next((lp for token, lp in top_logprobs.items() if token.strip().lower() == response.strip().lower()), None)
            for token, lp in top_logprobs.items():
                if token.strip().lower() == response.strip().lower():
                    matching_logprob = lp
                    parsed_responses.append(token)
                    break
            logprobs[response] = matching_logprob if matching_logprob is not None else float('-inf')

        idx = max(range(len(responses)), key=lambda i: logprobs[responses[i]])
        max_str = responses[idx]

        print(f"Validation: responses={responses}, logprobs={logprobs}, parsed_responses={parsed_responses}, selected_idx={idx}, selected_response={max_str}")

        return idx, max_str, logprobs






