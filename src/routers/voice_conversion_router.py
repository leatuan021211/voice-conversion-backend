from logging import getLogger
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
import os
import shutil

from core.settings import LoggerSettings
from services.voice_conversion_service import VoiceConversionService
from core.messages import ResponeMessages

logger = getLogger(__name__)
logger.setLevel(LoggerSettings.LOG_LEVEL)

router = APIRouter(prefix="/api/voice_conversion", tags=["Voice Conversion"])

@router.post("/")
async def voice_conversion(source_audio: UploadFile = File(...), target_audio: UploadFile = File(...)):
    # Create temporary files to store uploaded files
    temp_source = None
    temp_target = None
    
    try:
        # Validate file extensions
        valid_extensions = ['.wav', '.mp3', '.ogg', '.flac']
        
        # Get file extension from source audio file
        source_ext = os.path.splitext(source_audio.filename)[1].lower()
        if not source_ext or not any(source_ext == ext for ext in valid_extensions):
            return JSONResponse(
                status_code=400,
                content={"message": f"Source audio file must have one of these extensions: {', '.join(valid_extensions)}"}
            )
            
        # Get file extension from target audio file
        target_ext = os.path.splitext(target_audio.filename)[1].lower()
        if not target_ext or not any(target_ext == ext for ext in valid_extensions):
            return JSONResponse(
                status_code=400,
                content={"message": f"Target audio file must have one of these extensions: {', '.join(valid_extensions)}"}
            )
        
        # Save uploaded files to temporary files with correct extensions
        temp_source = tempfile.NamedTemporaryFile(delete=False, suffix=source_ext)
        temp_target = tempfile.NamedTemporaryFile(delete=False, suffix=target_ext)
        
        # Write uploaded content to temporary files
        shutil.copyfileobj(source_audio.file, temp_source)
        shutil.copyfileobj(target_audio.file, temp_target)
        
        # Close the files to ensure all data is written
        temp_source.close()
        temp_target.close()
        
        # Get file paths
        source_path = temp_source.name
        target_path = temp_target.name
        
        # Reset file positions
        source_audio.file.seek(0)
        target_audio.file.seek(0)
        
        # Call service with file paths instead of file objects
        audio_base64, text = VoiceConversionService.converse_voice_from_paths(source_path, target_path)
        
        return JSONResponse(
            content={
                "audio_base64": audio_base64,
                "text": text,
                "message": "Voice conversion completed successfully."
            }
        )
    except Exception as e:
        logger.error(f"Error in voice conversion: {e}")
        return JSONResponse(
            status_code=500, 
            content={"message": f"{ResponeMessages.ERROR_500}. Details: {str(e)}"}
        )
    finally:
        # Clean up temporary files
        if temp_source and os.path.exists(temp_source.name):
            os.unlink(temp_source.name)
        if temp_target and os.path.exists(temp_target.name):
            os.unlink(temp_target.name)