import unittest
from typing import Type
import asyncio

from g4f.models import __models__
from g4f.providers.base_provider import BaseProvider, ProviderModelMixin
from g4f.models import Model

class TestProviderHasModel(unittest.IsolatedAsyncioTestCase):
    cache: dict = {}

    async def test_provider_has_model(self):
        for model, providers in __models__.values():
            for provider in providers:
                if issubclass(provider, ProviderModelMixin):
                    if model.name in provider.model_aliases:
                        model_name = provider.model_aliases[model.name]
                    else:
                        model_name = model.name
                    await asyncio.wait_for(self.provider_has_model(provider, model_name), 10)

    async def provider_has_model(self, provider: Type[BaseProvider], model: str):
        if provider.__name__ not in self.cache:
            self.cache[provider.__name__] = provider.get_models()
        if self.cache[provider.__name__]:
            self.assertIn(model, self.cache[provider.__name__], provider.__name__)