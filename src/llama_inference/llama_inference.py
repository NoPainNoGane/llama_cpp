from llama_cpp import Llama
from env import SYSTEM_PROMPT, DEFAULT_JSON, MODEL_PATH
import logging
# Настраиваем логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Llama_cpp:
    def __init__(self, model_name="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf") -> None:
        logger.info("Инициализация модели LLaMA...")
        self.model = Llama(
            model_path=f'{MODEL_PATH}{model_name}',
            n_ctx=8192,
            n_parts=1,
            verbose=True,
            n_threads=7,
            n_gpu_layers = -1
        )
        logger.info(f"Модель загружена с {MODEL_PATH}{model_name}.")

    async def generate(
                        self,
                        message: str, 
                        response_format = DEFAULT_JSON, 
                        system_prompt: str = SYSTEM_PROMPT,
                        temperature: float = 0.0,
                        top_k: int = 50,
                        top_p: float = 0.85,
                        repeat_penalty: float = 1.2,
                    ) -> str:
        logger.info("Начало генерации текста с использованием LLaMA...")
        messages = [{
            "role": "system", 
            "content": system_prompt
        }]

        messages.append({
            "role": "user", 
            "content": message
        })
        
        response = ""
        for part in self.model.create_chat_completion(
                messages,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=float(repeat_penalty),
                stream=True,
                response_format=response_format
            ):
                delta = part["choices"][0]["delta"]
                if "content" in delta:
                    response += delta["content"]
                    
        return response 