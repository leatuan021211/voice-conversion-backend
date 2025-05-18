from logging import getLogger
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from core.settings import LoggerSettings
from services.speaker_verification_service import SpeakerVerificationService
from core.messages import ResponeMessages

logger = getLogger(__name__)
logger.setLevel(LoggerSettings.LOG_LEVEL)

router = APIRouter(prefix="/api/speaker_verification", tags=["Speaker Verification"])

@router.post("/similarity")
async def get_similarity(audio1: UploadFile = File(...), aduio2: UploadFile = File(...)):
    """
    Endpoint to compare two audi1o files and return their similarity score.
    
    Args:
        audio1 (UploadFile): First audio file.
        aduio2 (UploadFile): Second audio file.
    
    Returns:
        JSONResponse: Similarity score between the two audio files.
    """
    try:
        similarity = SpeakerVerificationService.cal_cosine_similarity(audio1.file, aduio2.file)
        return JSONResponse(
            content={
                "similarity": similarity.item(),
                "message": "Similarity calculated successfully."
                }
            )
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return JSONResponse(status_code=500, content={"message": ResponeMessages.ERROR_500})