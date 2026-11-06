# AsyncLLMManager

Capa de acceso unificada y asíncrona a modelos de lenguaje. Permite usar **OpenAI** o **Anthropic** con la misma interfaz, validar los parámetros de entrada con **Pydantic**, recibir respuestas en **streaming** y manejar errores de red y límite de tasa sin que el programa se caiga.

## Características

- **Multi-proveedor**: elegí OpenAI o Anthropic cambiando una sola variable en el `.env`.
- **Validación de parámetros**: `LLMParams` (Pydantic) rechaza valores inválidos antes de llamar a la API (por ejemplo, una temperatura fuera del rango 0 a 2).
- **Asíncrono**: construido sobre `asyncio` para atender varias consultas en paralelo sin bloquearse.
- **Streaming**: `generate_stream()` entrega el texto en fragmentos a medida que llega, usando `yield`.
- **Errores controlados**: los fallos de red, el límite de tasa y la falta de créditos se devuelven como un objeto `LLMError` en lugar de lanzar una excepción.

## Requisitos

- Python 3.12 o superior
- Una clave de API de OpenAI y/o de Anthropic, con saldo disponible

## Instalación

1. Cloná o descargá el proyecto y entrá a la carpeta:

   ```bash
   cd Pre-entrega-1
   ```

2. Creá y activá un entorno virtual:

   ```bash
   python -m venv venv
   ```

   - Windows: `venv\Scripts\activate`
   - macOS / Linux: `source venv/bin/activate`

3. Instalá las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

## Configuración

Creá un archivo `.env` en la raíz del proyecto con este contenido:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=tu_clave_de_openai
ANTHROPIC_API_KEY=tu_clave_de_anthropic
```

| Variable | Descripción |
|---|---|
| `LLM_PROVIDER` | Proveedor a usar: `openai` o `anthropic`. Por defecto, `openai`. |
| `OPENAI_API_KEY` | Clave de la API de OpenAI (solo si usás `openai`). |
| `ANTHROPIC_API_KEY` | Clave de la API de Anthropic (solo si usás `anthropic`). |

> **Importante:** no subas el archivo `.env` a un repositorio. Agregalo a tu `.gitignore`.

## Uso

### Generación completa

```python
import asyncio
from llm_manager import AsyncLLMManager, LLMError

async def main():
    manager = AsyncLLMManager()  # usa LLM_PROVIDER del .env
    resultado = await manager.generate(
        "Decime hola en una frase.",
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=100,
    )

    if isinstance(resultado, LLMError):
        print(f"Error ({resultado.tipo}): {resultado.mensaje}")
    else:
        print(resultado)

asyncio.run(main())
```

### Streaming

```python
async def main_stream():
    manager = AsyncLLMManager()
    async for fragmento in manager.generate_stream(
        "Contame un chiste corto.",
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=200,
    ):
        if isinstance(fragmento, LLMError):
            print(f"\nError ({fragmento.tipo}): {fragmento.mensaje}")
        else:
            print(fragmento, end="", flush=True)

asyncio.run(main_stream())
```

### Elegir el proveedor desde el código

```python
manager = AsyncLLMManager(provider="anthropic")
```

Con Anthropic, usá un modelo de Claude, por ejemplo `model="claude-haiku-5-5"`.

## Parámetros

Los parámetros se validan con la clase `LLMParams` definida en `schemas.py`. Se pasan como argumentos con nombre a `generate()` y `generate_stream()`.

| Parámetro | Tipo | Por defecto | Restricción |
|---|---|---|---|
| `model` | `str` | *(obligatorio)* | No puede estar vacío; se eliminan los espacios sobrantes |
| `temperature` | `float` | `1.0` | Entre 0 y 2 |
| `max_tokens` | `int` | `1024` | Mayor que 0 |
| `top_p` | `float` | `1.0` | Mayor que 0 y hasta 1 |
| `frequency_penalty` | `float` | `0.0` | Entre -2 y 2 |
| `presence_penalty` | `float` | `0.0` | Entre -2 y 2 |
| `stop` | `list[str]` o `None` | `None` | Secuencias que detienen la generación |
| `stream` | `bool` | `False` | Reservado: para streaming se usa `generate_stream()` |

Si algún valor es inválido, o si se pasa un parámetro que no existe en el esquema (`extra="forbid"`), se lanza un `ValidationError` de Pydantic **antes** de hacer la llamada a la API.

> **Notas sobre Anthropic:**
> - Su API acepta temperaturas de 0 a 1, así que valores mayores se limitan automáticamente a 1.0.
> - `top_p`, `frequency_penalty` y `presence_penalty` no se envían a Anthropic.
> - `stop` se traduce al parámetro `stop_sequences` de su API.

## Manejo de errores

Los fallos del entorno no lanzan excepciones: se devuelve un `LLMError` con un `tipo` y un `mensaje`. En streaming, el error se entrega como último elemento del generador, ya que puede ocurrir a mitad de la respuesta.

| Tipo | Cuándo ocurre |
|---|---|
| `rate_limit` | Se superó el límite de solicitudes por tiempo |
| `sin_credito` | La cuenta de OpenAI no tiene saldo disponible |
| `red` | Falló la conexión o se agotó el tiempo de espera |
| `api` | Otro error HTTP de la API (clave inválida, error del servidor, etc.) |

## Estructura del proyecto

```
Pre-entrega-1/
├── llm_manager.py     # Clase AsyncLLMManager y LLMError
├── schemas.py         # LLMParams: validación de parámetros con Pydantic
├── requirements.txt   # Dependencias del proyecto
├── README.md          # Este archivo
├── .gitignore         # Archivos que Git debe ignorar
├── .env               # Claves y proveedor (no se sube al repositorio)
└── venv/              # Entorno virtual (no se sube al repositorio)
```

## Ejecutar la prueba incluida

`llm_manager.py` incluye un ejemplo al final del archivo que prueba la generación completa y el streaming:

```bash
python llm_manager.py
```

## Tecnologías

- [Python](https://www.python.org/)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Pydantic](https://docs.pydantic.dev/)
- [python-dotenv](https://github.com/theskumar/python-dotenv)
