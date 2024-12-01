from fastapi import APIRouter
from llama_inference.llama_inference import Llama_cpp
from env import SYSTEM_PROMPT, DEFAULT_JSON, MODEL_PATH
from pydantic import BaseModel
from typing import Optional

llama_router = APIRouter()

# Определите Pydantic модель для параметров
class LlamaCppRequest(BaseModel):
    message: str
    model_name: Optional[str] = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    response_format: str = DEFAULT_JSON
    system_prompt: str = SYSTEM_PROMPT
    temperature: float = 0.0
    top_k: int = 50
    top_p: float = 0.85
    repeat_penalty: float = 1.2

class Llama_cpp_controller:

    @llama_router.post(path="/llamacpp")
    async def llama_cpp(request: LlamaCppRequest) -> str:
        service = Llama_cpp(model_name=request.model_name)
        return await service.generate(
                                        message = request.message,
                                        response_format = request.response_format,
                                        system_prompt = request.system_prompt,
                                        temperature = request.temperature,
                                        top_k = request.top_k,
                                        top_p = request.top_p,
                                        repeat_penalty = request.repeat_penalty
                                    )