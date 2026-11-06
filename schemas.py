# schemas.py
from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., min_length=1, description="Nombre del modelo")
    temperature: float = Field(default=1.0, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    frequency_penalty: float = Field(default=0.0, ge=-2, le=2)
    presence_penalty: float = Field(default=0.0, ge=-2, le=2)
    stop: list[str] | None = None
    stream: bool = False

    @field_validator("model")
    @classmethod
    def model_sin_espacios(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("El nombre del modelo no puede estar vacío")
        return v