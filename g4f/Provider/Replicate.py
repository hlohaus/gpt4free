from __future__ import annotations

import asyncio

from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt, filter_none, get_cookies
from ..typing import AsyncResult, Messages, ImageType
from ..image import to_data_uri
from ..requests import raise_for_status
from ..requests.aiohttp import StreamSession
from ..errors import ResponseError, MissingAuthError

class Replicate(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://replicate.com"
    working = True
    needs_auth = True
    default_model = "meta/meta-llama-3-70b-instruct"
    default_vision_model = "yorickvp/llava-v1.6-34b"
    models = [
        default_model,
        "meta/meta-llama-3-8b-instruct",
        "mistralai/mixtral-8x7b-instruct-v0.1",
        default_vision_model,
        "daanelson/minigpt-4",
    ]
    model_aliases = {
        "meta-llama/Meta-Llama-3-70B-Instruct": default_model,
        "meta-llama/Meta-Llama-3-8B-Instruct": "meta/meta-llama-3-8b-instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "mistralai/mixtral-8x7b-instruct-v0.1",
    }
    minigpt_4_model = "daanelson/minigpt-4"
    versions = {
        default_vision_model: "41ecfbfb261e6c1adf3ad896c9066ca98346996d7c4045c5bc944a79d430f174",
        minigpt_4_model: "b96a2f33cc8e4b0aa23eacfce731b9c41a7d9466d9ed4e167375587b54db9423"
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        api_key: str = None,
        proxy: str = None,
        timeout: int = 180,
        version: str = None,
        image: ImageType = None,
        system_prompt: str = None,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        top_k: float = None,
        stop: list = None,
        extra_data: dict = {},
        headers: dict = {
            "Accept": "application/json",
        },
        **kwargs
    ) -> AsyncResult:
        if api_key is None:
            cookies = get_cookies("replicate.com")
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
                "Accept": "application/json",
                "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://replicate.com/",
                "Content-Type": "application/json",
                "X-CSRFToken": cookies.get("csrftoken"),
                "X-REPLICATE-BLOG-ANALYTICS-ID:": f"run-llama-3-with-an-api-{cookies.get('replicate_anonymous_id')}",
                "Origin": "https://replicate.com",
                "Connection": "keep-alive",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                **headers
            }
        if not model and image is not None:
            model = cls.get_model(cls.default_vision_model)
        elif not model or model in cls.model_aliases:
            model = cls.get_model(model)
        if model == cls.minigpt_4_model:
            stream = False
        if cls.needs_auth and api_key is None:
            raise MissingAuthError('Add a "api_key"')
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
            api_base = "https://api.replicate.com/v1/models/"
        else:
            api_base = "https://replicate.com/api/models/"
        if model in cls.versions:
            url = "https://api.replicate.com/v1/predictions"
        elif version is None:
            url = f"{api_base.rstrip('/')}/{model}/predictions"
        else:
            url = f"{api_base.rstrip('/')}/{model}/versions/{version}/predictions"
        async with StreamSession(
            proxy=proxy,
            headers=headers,
            timeout=timeout,
            cookies=cookies
        ) as session:
            data = {
                "stream": stream,
                "version": cls.versions[model] if model in cls.versions else version,
                "input": {
                    "prompt": format_prompt(messages),
                    **filter_none(
                        image=None if image is None else to_data_uri(image),
                        system_prompt=system_prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        stop_sequences=",".join(stop) if stop else None
                    ),
                    **extra_data
                },
            }
            async with session.post(url, json=data) as response:
                message = "Model not found" if response.status == 404 else None
                await raise_for_status(response, message)
                result = await response.json()
                if "id" not in result:
                    raise ResponseError(f"Invalid response: {result}")
                if not stream:
                    while result["status"] in ("starting", "processing"):
                        await asyncio.sleep(0.1)
                        async with session.get(result["urls"]["get"]) as response:
                            await raise_for_status(response)
                            result = await response.json()
                    yield result["output"]
                    return
                async with session.get(result["urls"]["stream"], headers={"Accept": "text/event-stream"}) as response:
                    await raise_for_status(response)
                    event = None
                    async for line in response.iter_lines():
                        if line.startswith(b"event: "):
                            event = line[7:]
                            if event == b"done":
                                break
                        elif event == b"output":
                            if line.startswith(b"data: "):
                                new_text = line[6:].decode()
                                if new_text:
                                    yield new_text
                                else:
                                    yield "\n"