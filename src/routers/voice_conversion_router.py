from logging import getLogger
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
import os
import shutil

from core.settings import LoggerSettings
from services.voice_conversion_service import VoiceConversionService

logger = getLogger(__name__)
logger.setLevel(LoggerSettings.LOG_LEVEL)

router = APIRouter(prefix="/api/voice_conversion", tags=["Voice Conversion"])

@router.post("/")
async def voice_conversion(source: UploadFile = File(...), voice_name: str = "wahahaha"):
    audio_base64 = VoiceConversionService.converse_voice(source.file, voice_name)
    return JSONResponse(
        content={
            "audio_base64": audio_base64,
            "message": "Voice conversion completed successfully."
        }
    )


@router.get("/voices")
def list_voices():
    """
    Endpoint to list all available voices.
    
    Returns:
        JSONResponse: List of available voices.
    """
    try:
        voices = VoiceConversionService.get_available_voices()
        return JSONResponse(
            content={
                "voices": voices,
                "message": "Voices retrieved successfully."
            }
        )
    except Exception as e:
        logger.error(f"Error retrieving voices: {e}")
        return JSONResponse(status_code=500, content={"message": "Internal Server Error"})
