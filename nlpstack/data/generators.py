import asyncio
import json
import os
from contextlib import suppress
from os import PathLike
from typing import Any, List, Literal, Optional, Sequence, Type, TypedDict, Union

import minato
import requests

from nlpstack.common import cached_property
from nlpstack.transformers import cache as transformers_cache

try:
    import transformers
except ModuleNotFoundError:
    transformers = None

try:
    import openai
except ModuleNotFoundError:
    openai = None  # type: ignore[assignment]


class TextGenerator:
    """
    A base class for text generators.
    Text generators are callable objects that take a list of strings as input and return a list of strings as output.
    """

    def __call__(self, inputs: Sequence[str], **kwargs: Any) -> List[str]:
        """
        Generate text from a list of strings.

        Args:
            inputs: A list of prompt/context strings.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of generated strings.
        """
        raise NotImplementedError


class OpenAIChatTextGenerator(TextGenerator):
    """
    A text generator using OpenAI chat completion API.

    You need to set `OPENAI_API_KEY` environment variable to use this class.

    Args:
        model_name: The name of the model to use. Defaults to `"gpt-3.5-turbo"`.
        context: The context to use. Defaults to `None`.
        max_retries: The maximum number of retries when the API returns an error. Defaults to `3`.
        retry_interval: The interval between retries in seconds. Defaults to `10.0`.
        **kwargs: Additional keyword arguments to pass to the API.
    """

    Role = Literal["system", "assistant", "user"]

    class Message(TypedDict):
        role: "OpenAIChatTextGenerator.Role"
        content: str

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        context: Optional[Union[str, Sequence[Message]]] = None,
        max_retries: int = 3,
        retry_interval: float = 10.0,
        **kwargs: Any,
    ) -> None:
        if openai is None:
            raise ModuleNotFoundError(
                "OpenAI API is not installed. Please make sure `openai` is successfully installed."
            )

        if isinstance(context, str):
            context = [{"role": "system", "content": context}]

        self._model_name = model_name
        self._context = context
        self._max_retries = max_retries
        self._retry_interval = retry_interval
        self._kwargs = kwargs

        self._client

    @cached_property
    def _client(self) -> Type["openai.ChatCompletion"]:
        assert openai is not None
        return openai.ChatCompletion

    def __call__(self, inputs: Sequence[str], **kwargs: Any) -> List[str]:
        async def task(text: str) -> str:
            for _ in range(self._max_retries):
                response = await self._client.acreate(
                    model=self._model_name,
                    messages=list(self._context or []) + [{"role": "user", "content": text}],
                    **self._kwargs,
                    **kwargs,
                )
                if response.status_code == 200:
                    return str(response["choices"][0]["message"]["content"])
                elif 500 <= response.status_code < 600:
                    await asyncio.sleep(self._retry_interval)
                    continue
            response.raise_for_status()
            return str(response["choices"][0]["message"]["content"])

        async def main() -> List[str]:
            return await asyncio.gather(*[task(text) for text in inputs])

        return asyncio.run(main())


class HuggingfaceTextGenerator(TextGenerator):
    """
    A text generator using Huggingface inference API.
    This generator only supports text/text2text generation models such as `gpt2`.

    Args:
        model_name: The name of the model to use. Defaults to `"gpt2"`.
        max_retries: The maximum number of retries when the API returns an error. Defaults to `3`.
        retry_interval: The interval between retries in seconds. Defaults to `10.0`.
        **kwargs: Additional keyword arguments to pass to the API.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        max_retries: int = 3,
        retry_interval: float = 10.0,
        **kwargs: Any,
    ) -> None:
        self._api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self._max_retries = max_retries
        self._retry_interval = retry_interval
        self._kwargs = kwargs
        self._session

    @cached_property
    def _session(self) -> requests.Session:
        api_key = os.environ.get("HUGGINGFACE_API_KEY")
        if api_key is None:
            raise ValueError("Please provide an HuggingFace API key.")
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {api_key}"})
        return session

    def __call__(self, inputs: Sequence[str], **kwargs: Any) -> List[str]:
        async def task(text: str) -> str:
            data = json.dumps({"inputs": text, "parameters": {**self._kwargs, **kwargs}})
            loop = asyncio.get_running_loop()
            for _ in range(self._max_retries):
                response = await loop.run_in_executor(None, self._session.post, self._api_url, data)
                if response.status_code == 200:
                    return str(response.json()[0]["generated_text"])
                elif 500 <= response.status_code < 600:
                    await asyncio.sleep(self._retry_interval)
                    continue
            response.raise_for_status()
            return str(response.json()[0]["generated_text"])

        async def main() -> List[str]:
            return await asyncio.gather(*[task(text) for text in inputs])

        return asyncio.run(main())


class PretrainedTransformerTextGenerator(TextGenerator):
    """
    A text generator using pretrained transformer models.

    Args:
        model_name: The name of the model to use. Defaults to `"gpt2"`.
        task: The task to perform. Defaults to `None`.
        device: The device to use. Defaults to `"cpu"`.
        **kwargs: Additional keyword arguments to pass to the model.
    """

    Task = Literal["text-generation", "text2text-generation"]

    def __init__(
        self,
        pretrained_model_name: Union[str, PathLike] = "gpt2",
        task: Optional[Task] = None,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        self._pretrained_model_name = pretrained_model_name
        self._task = task
        self._device = device
        self._kwargs = kwargs

        self._pipeline = transformers.pipeline(
            task=task,
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            **kwargs,
        )

    @cached_property
    def tokenizer(self) -> "transformers.PreTrainedTokenizer":
        pretrained_model_name = self._pretrained_model_name
        with suppress(FileNotFoundError):
            pretrained_model_name = minato.cached_path(pretrained_model_name)
        return transformers_cache.get_pretrained_tokenizer(pretrained_model_name)

    @cached_property
    def model(self) -> "transformers.PreTrainedModel":
        pretrained_model_name = self._pretrained_model_name
        with suppress(FileNotFoundError):
            pretrained_model_name = minato.cached_path(pretrained_model_name)
        auto_cls = transformers.pipelines.SUPPORTED_TASKS[self._task]["pt"][0]
        model = transformers_cache.get_pretrained_model(pretrained_model_name, auto_cls=auto_cls)
        return model

    def __call__(self, inputs: Sequence[str], **kwargs: Any) -> List[str]:
        output = self._pipeline(
            inputs,
            **self._kwargs,
            **kwargs,
        )
        return [text[0]["generated_text"] for text in output]
