from logging import getLogger
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from core.settings import LoggerSettings
from services.voice_conversion_service import VoiceConversionService
from core.messages import ResponeMessages

logger = getLogger(__name__)
logger.setLevel(LoggerSettings.LOG_LEVEL)

router = APIRouter(prefix="/api/voice_conversion", tags=["Voice Conversion"])

@router.post("/")
async def voice_conversion(source: UploadFile = File(...), target: UploadFile = File(...)):
    mel = VoiceConversionService.converse_voice(source.file, target.file)
    return JSONResponse(
        content={
            "mel": mel,
            "message": "Voice conversion completed successfully."
        }
    )