from pathlib import Path

MODEL_PATH = f"{str(Path(__file__).resolve().parent)}/weights/"

SYSTEM_PROMPT = '''Создай JSON из текста. Если поле не названо или его значение отсутствует, 
                установи его как `null`. Обязательно все поля должны быть в исходном JSON! 
                Используй только полученный JSON, без дополнительных символов и текстов. 
                Если есть фраза: нормальный режим работы, то запиши ее в поле оценка шума иначе напиши туда тип шума из текста. 
                Текст который не удалось отнести к полям вставь в поле Примечание. 
                В ответе должен быть только один уже отредактированный JSON. '''
                
DEFAULT_JSON = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "НГДУ": {
                "type": "string",
                "description": "НГДУ."
            },
            "цех": {
                "type": "string",
                "description": "цех"
            },
            "номер скважины": {
                "type": "string",
                "description": "номер скважины"
            },
            "тип привода": {
                "type": "string",
                "description": "тип привода"
            },
            "марка привода": {
                "type": "string",
                "description": "марка привода",
                "default": None
            },
            "заявка в АРМИТС": {
                "type": "integer",
                "description": "заявка в Армитс",
                "default": None
            },
            "тип шума": {
                "type": "string",
                "description": "тип шума",
                "default": None
            },
            "оценка шума": {
                "type": "string",
                "description": "экспертная оценка шума (что шумит)",
                "default": None
            },
            "Примечание": {
                "type": "string",
                "description": "Примечание",
                "default": None
            }
        },
        "required": [
            "НГДУ",
            "цех",
            "номер скважины",
            "тип привода",
            "марка привода",
            "заявка в АРМИТС"
        ]
    },
    "required": ["items"]
    }