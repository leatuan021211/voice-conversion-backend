from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from routers.speaker_verification_router import router as speaker_verification_router
from routers.voice_conversion_router import router as voice_conversion_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(speaker_verification_router)
app.include_router(voice_conversion_router)