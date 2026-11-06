# llm_manager.py
import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

import anthropic
import openai
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI

from schemas import LLMParams

load_dotenv()

# El orden importa al capturar: RateLimitError y APITimeoutError son
# subclases de APIStatusError y APIConnectionError respectivamente.
RATE_LIMIT_ERRORS = (openai.RateLimitError, anthropic.RateLimitError)
NETWORK_ERRORS = (openai.APIConnectionError, anthropic.APIConnectionError)
API_ERRORS = (openai.APIStatusError, anthropic.APIStatusError)
HANDLED_ERRORS = RATE_LIMIT_ERRORS + NETWORK_ERRORS + API_ERRORS


@dataclass
class LLMError:
    """Error controlado: se devuelve (o se yieldea) en lugar de lanzar una excepción."""

    tipo: str  # "rate_limit", "sin_credito", "red" o "api"
    mensaje: str


class AsyncLLMManager:
    def __init__(self, provider: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
        self.client = self._load_client()

    def _load_client(self):
        if self.provider == "openai":
            return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if self.provider == "anthropic":
            return AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        raise ValueError(
            f"Proveedor no soportado: '{self.provider}'. Usá 'openai' o 'anthropic'."
        )

    # ------------------------------------------------------------------
    # Traducción de excepciones a un error controlado
    # ------------------------------------------------------------------
    def _to_llm_error(self, exc: Exception) -> LLMError:
        if isinstance(exc, RATE_LIMIT_ERRORS):
            # OpenAI usa 429 también cuando te quedás sin saldo
            if getattr(exc, "code", None) == "insufficient_quota":
                return LLMError(
                    "sin_credito",
                    f"[{self.provider}] No hay créditos disponibles en la cuenta.",
                )
            return LLMError(
                "rate_limit",
                f"[{self.provider}] Límite de tasa alcanzado. Esperá un momento y reintentá.",
            )
        if isinstance(exc, NETWORK_ERRORS):
            return LLMError(
                "red",
                f"[{self.provider}] No se pudo conectar con la API (red o timeout).",
            )
        # Cualquier otro error HTTP de la API (401, 400, 500, ...)
        status = getattr(exc, "status_code", "?")
        return LLMError("api", f"[{self.provider}] Error de la API (HTTP {status}).")

    # ------------------------------------------------------------------
    # Generación completa (sin streaming)
    # ------------------------------------------------------------------
    async def generate(self, prompt: str, **kwargs) -> str | LLMError:
        params = LLMParams(**kwargs)  # valida; si algo está mal, lanza ValidationError

        try:
            if self.provider == "openai":
                return await self._generate_openai(prompt, params)
            return await self._generate_anthropic(prompt, params)
        except HANDLED_ERRORS as exc:
            return self._to_llm_error(exc)

    async def _generate_openai(self, prompt: str, params: LLMParams) -> str:
        response = await self.client.chat.completions.create(
            model=params.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=params.temperature,
            max_completion_tokens=params.max_tokens,
            top_p=params.top_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
            stop=params.stop,
        )
        return response.choices[0].message.content

    async def _generate_anthropic(self, prompt: str, params: LLMParams) -> str:
        request = {
            "model": params.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": min(params.temperature, 1.0),  # Anthropic solo llega a 1
            "max_tokens": params.max_tokens,
        }
        if params.stop:
            request["stop_sequences"] = params.stop

        response = await self.client.messages.create(**request)
        return response.content[0].text

    # ------------------------------------------------------------------
    # Generación con streaming
    # ------------------------------------------------------------------
    async def generate_stream(
        self, prompt: str, **kwargs
    ) -> AsyncIterator[str | LLMError]:
        params = LLMParams(**kwargs)  # misma validación que generate()

        try:
            # En generadores asíncronos no existe `yield from`: hay que re-emitir.
            if self.provider == "openai":
                async for fragmento in self._stream_openai(prompt, params):
                    yield fragmento
            else:
                async for fragmento in self._stream_anthropic(prompt, params):
                    yield fragmento
        except HANDLED_ERRORS as exc:
            # El error puede llegar a mitad del stream: se entrega como último elemento
            yield self._to_llm_error(exc)

    async def _stream_openai(
        self, prompt: str, params: LLMParams
    ) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=params.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=params.temperature,
            max_completion_tokens=params.max_tokens,
            top_p=params.top_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
            stop=params.stop,
            stream=True,
        )
        async for chunk in stream:
            # Algunos chunks vienen sin texto (el primero, el último)
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _stream_anthropic(
        self, prompt: str, params: LLMParams
    ) -> AsyncIterator[str]:
        request = {
            "model": params.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": min(params.temperature, 1.0),
            "max_tokens": params.max_tokens,
        }
        if params.stop:
            request["stop_sequences"] = params.stop

        async with self.client.messages.stream(**request) as stream:
            async for texto in stream.text_stream:
                yield texto


# ----------------------------------------------------------------------
# Pruebas
# ----------------------------------------------------------------------
def _modelo_para(manager: AsyncLLMManager) -> str:
    return "gpt-4o-mini" if manager.provider == "openai" else "claude-haiku-5-5"


async def main():
    manager = AsyncLLMManager()
    resultado = await manager.generate(
        "Decime hola en una frase.",
        model=_modelo_para(manager),
        temperature=0.7,
        max_tokens=100,
    )
    if isinstance(resultado, LLMError):
        print(f"Error controlado ({resultado.tipo}): {resultado.mensaje}")
    else:
        print(resultado)


async def main_stream():
    manager = AsyncLLMManager()
    async for fragmento in manager.generate_stream(
        "Contame un chiste corto.",
        model=_modelo_para(manager),
        temperature=0.7,
        max_tokens=200,
    ):
        if isinstance(fragmento, LLMError):
            print(f"\nError controlado ({fragmento.tipo}): {fragmento.mensaje}")
        else:
            print(fragmento, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(main_stream())