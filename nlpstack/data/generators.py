import asyncio
import json
import os
from contextlib import suppress
from os import PathLike
from typing import Any, Generic, List, Literal, Mapping, NamedTuple, Optional, Sequence, TypedDict, TypeVar, Union

import minato
import requests

from nlpstack.common import Pipeline, cached_property

try:
    import transformers
except ModuleNotFoundError:
    transformers = None

try:
    import openai
except ModuleNotFoundError:
    openai = None  # type: ignore[assignment]


_T_Fixtures = TypeVar("_T_Fixtures")
_T_Params = TypeVar("_T_Params")


class TextGenerator(
    Pipeline[str, str, _T_Fixtures, Optional[_T_Params]],
    Generic[_T_Fixtures, _T_Params],
):
    """
    A base class for text generators.
    Text generators are callable objects that take a list of strings as input and return a list of strings as output.
    """


class OpenAIChatTextGenerator(
    TextGenerator[
        "OpenAIChatTextGenerator.Fixtures",
        Optional["OpenAIChatTextGenerator.Params"],
    ]
):
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

    class Fixtures(NamedTuple):
        client: "openai.AsyncOpenAI"

    class Params(TypedDict, total=False):
        frequency_penalty: float
        max_tokens: int
        temperature: float
        top_p: float

    def __init__(
        self,
        *,
        model_name: str = "gpt-3.5-turbo",
        context: Optional[Union[str, Sequence["openai.types.chat.ChatCompletionMessageParam"]]] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        batch_size: int = 1,
        max_workers: int = 1,
        **kwargs: Any,
    ) -> None:
        if max_workers > 1:
            raise ValueError("OpenAIChatTextGenerator does not support parallel processing.")
        super().__init__(batch_size=batch_size, max_workers=max_workers)
        if openai is None:
            raise ModuleNotFoundError(
                "OpenAI API is not installed. Please make sure `openai` is successfully installed."
            )

        if isinstance(context, str):
            context = [{"role": "system", "content": context}]

        self._model_name = model_name
        self._organization = organization
        self._base_url = base_url
        self._api_key = api_key
        self._context = context
        self._max_retries = max_retries
        self._kwargs = kwargs

    def get_client(self) -> "openai.AsyncOpenAI":
        assert openai is not None
        return openai.AsyncOpenAI(
            organization=self._organization,
            base_url=self._base_url,
            api_key=self._api_key,
            max_retries=self._max_retries,
        )

    @cached_property
    def fixtures(self) -> "OpenAIChatTextGenerator.Fixtures":  # type: ignore[override]
        return OpenAIChatTextGenerator.Fixtures(self.get_client())

    def apply_batch(
        self,
        batch: Sequence[str],
        fixtures: "OpenAIChatTextGenerator.Fixtures",
        params: Optional["OpenAIChatTextGenerator.Params"] = None,
    ) -> List[str]:
        async def task(text: str) -> str:
            response = await fixtures.client.chat.completions.create(
                model=self._model_name,
                messages=list(self._context or []) + [{"role": "user", "content": text}],
                **self._kwargs,
                **(params or {}),
            )
            return str(response.choices[0].message.content)

        async def main() -> List[str]:
            return await asyncio.gather(*[task(text) for text in batch])

        return asyncio.run(main())


class HuggingfaceTextGenerator(TextGenerator["HuggingfaceTextGenerator.Fixtures", None]):
    """
    A text generator using Huggingface inference API.
    This generator only supports text/text2text generation models such as `gpt2`.

    Args:
        model_name: The name of the model to use. Defaults to `"gpt2"`.
        max_retries: The maximum number of retries when the API returns an error. Defaults to `3`.
        retry_interval: The interval between retries in seconds. Defaults to `10.0`.
        **kwargs: Additional keyword arguments to pass to the API.
    """

    class Fixtures(NamedTuple):
        session: requests.Session

    def __init__(
        self,
        *,
        model_name: str = "gpt2",
        max_retries: int = 3,
        retry_interval: float = 10.0,
        batch_size: int = 1,
        max_workers: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(batch_size=batch_size, max_workers=max_workers)
        self._api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self._max_retries = max_retries
        self._retry_interval = retry_interval
        self._kwargs = kwargs

    def get_session(self) -> requests.Session:
        api_key = os.environ.get("HUGGINGFACE_API_KEY")
        if api_key is None:
            raise ValueError("Please provide an HuggingFace API key.")
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {api_key}"})
        return session

    @property
    def fixtures(self) -> "HuggingfaceTextGenerator.Fixtures":
        return HuggingfaceTextGenerator.Fixtures(self.get_session())

    def apply_batch(
        self,
        inputs: Sequence[str],
        fixtures: "HuggingfaceTextGenerator.Fixtures",
        params: None = None,
    ) -> List[str]:
        del params

        async def task(text: str) -> str:
            data = json.dumps({"inputs": text, "parameters": {**self._kwargs}})
            loop = asyncio.get_running_loop()
            for _ in range(self._max_retries):
                response = await loop.run_in_executor(None, fixtures.session.post, self._api_url, data)
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


class PretrainedTransformerTextGenerator(TextGenerator["PretrainedTransformerTextGenerator.Fixtures", None]):
    """
    A text generator using pretrained transformer models.

    Args:
        model_name: The name of the model to use. Defaults to `"gpt2"`.
        task: The task to perform. Defaults to `None`.
        device: The device to use. Defaults to `"cpu"`.
        **kwargs: Additional keyword arguments to pass to the model.
    """

    Task = Literal["text-generation", "text2text-generation"]

    class Fixtures(NamedTuple):
        pipeline: "transformers.Pipeline"
        kwargs: Mapping[str, Any]

    def __init__(
        self,
        *,
        pretrained_model_name: Union[str, PathLike] = "gpt2",
        task: Optional[Task] = None,
        device: str = "cpu",
        batch_size: int = 1,
        max_workers: int = 1,
        **kwargs: Any,
    ) -> None:
        if transformers is None:
            raise ModuleNotFoundError("transformers is not installed")

        super().__init__(batch_size=batch_size, max_workers=max_workers)
        self._pretrained_model_name = pretrained_model_name
        self._task = task
        self._device = device
        self._kwargs = kwargs

    def get_tokenizer(self) -> "transformers.PreTrainedTokenizer":
        from nlpstack.integrations.transformers import cache as transformers_cache

        pretrained_model_name = self._pretrained_model_name
        with suppress(FileNotFoundError):
            pretrained_model_name = minato.cached_path(pretrained_model_name)
        return transformers_cache.get_pretrained_tokenizer(pretrained_model_name)

    def get_model(self) -> "transformers.PreTrainedModel":
        from nlpstack.integrations.transformers import cache as transformers_cache

        pretrained_model_name = self._pretrained_model_name
        with suppress(FileNotFoundError):
            pretrained_model_name = minato.cached_path(pretrained_model_name)
        auto_cls = transformers.pipelines.SUPPORTED_TASKS[self._task]["pt"][0] if self._task else None
        model = transformers_cache.get_pretrained_model(pretrained_model_name, auto_cls=auto_cls)
        return model

    @cached_property
    def fixtures(self) -> "PretrainedTransformerTextGenerator.Fixtures":  # type: ignore[override]
        return PretrainedTransformerTextGenerator.Fixtures(
            transformers.pipeline(
                task=self._task,
                model=self.get_model() if self._task is not None else self._pretrained_model_name,
                tokenizer=self.get_tokenizer() if self._task is not None else None,
                device=self._device,
                **self._kwargs,
            ),
            self._kwargs,
        )

    def apply_batch(
        self,
        batch: Sequence[str],
        fixtures: "PretrainedTransformerTextGenerator.Fixtures",
        params: None = None,
    ) -> List[str]:
        output = fixtures.pipeline(
            list(batch),
            **fixtures.kwargs,
        )
        sequence_delimiter = {**fixtures.kwargs}.get("sequence_delimiter", "\n")
        return [
            (
                sequence_delimiter.join(x["generated_text"] for x in texts)
                if isinstance(texts, list)
                else texts["generated_text"]
            )
            for texts in output
        ]
