from __future__ import annotations

from urllib.parse import quote
import random
import requests
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from ..image import ImageResponse
from ..requests.raise_for_status import raise_for_status
from ..requests.aiohttp import get_connector
from .needs_auth.OpenaiAPI import OpenaiAPI
from .helper import format_prompt

class PollinationsAI(OpenaiAPI):
    label = "Pollinations.AI"
    url = "https://pollinations.ai"
    
    working = True
    needs_auth = False
    supports_stream = True
    
    default_model = "openai"
    
    additional_models_image = ["unity", "midijourney", "rtist"]
    additional_models_text = ["sur", "sur-mistral", "claude"]
    
    model_aliases = {
        "gpt-4o": "openai",
        "mistral-nemo": "mistral",
        "llama-3.1-70b": "llama", #
        "gpt-3.5-turbo": "searchgpt",
        "gpt-4": "searchgpt",
        "gpt-3.5-turbo": "claude",
        "gpt-4": "claude",
        "qwen-2.5-coder-32b": "qwen-coder", 
        "claude-3.5-sonnet": "sur", 
    }
    
    @classmethod
    def get_models(cls):
        if not hasattr(cls, 'image_models'):
            cls.image_models = []
        if not cls.image_models:
            url = "https://image.pollinations.ai/models"
            response = requests.get(url)
            raise_for_status(response)
            cls.image_models = response.json()
            cls.image_models.extend(cls.additional_models_image)
        if not hasattr(cls, 'models'):
            cls.models = []
        if not cls.models:
            url = "https://text.pollinations.ai/models"
            response = requests.get(url)
            raise_for_status(response)
            cls.models = [model.get("name") for model in response.json()]
            cls.models.extend(cls.image_models)
            cls.models.extend(cls.additional_models_text)
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        prompt: str = None,
        api_base: str = "https://text.pollinations.ai/openai",
        api_key: str = None,
        proxy: str = None,
        seed: str = None,
        width: int = 1024,
        height: int = 1024,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        if model in cls.image_models:
            async for response in cls._generate_image(model, messages, prompt, seed, width, height):
                yield response
        elif model in cls.models:
            async for response in cls._generate_text(model, messages, api_base, api_key, proxy, **kwargs):
                yield response
        else:
            raise ValueError(f"Unknown model: {model}")

    @classmethod
    async def _generate_image(cls, model: str, messages: Messages, prompt: str = None, seed: str = None, width: int = 1024, height: int = 1024):
        if prompt is None:
            prompt = messages[-1]["content"]
        if seed is None:
            seed = random.randint(0, 100000)
        image = f"https://image.pollinations.ai/prompt/{quote(prompt)}?width={width}&height={height}&seed={int(seed)}&nofeed=true&nologo=true&model={quote(model)}"
        yield ImageResponse(image, prompt)

    @classmethod
    async def _generate_text(cls, model: str, messages: Messages, api_base: str, api_key: str = None, proxy: str = None, **kwargs):
        if api_key is None:
            async with ClientSession(connector=get_connector(proxy=proxy)) as session:
                prompt = format_prompt(messages)
                async with session.get(f"https://text.pollinations.ai/{quote(prompt)}?model={quote(model)}") as response:
                    await raise_for_status(response)
                    async for line in response.content.iter_any():
                        yield line.decode(errors="ignore")
        else:
            async for chunk in super().create_async_generator(
                model, messages, api_base=api_base, proxy=proxy, **kwargs
            ):
                yield chunk