from pydantic import BaseModel

class RootResponse(BaseModel):
    project: str
    status: str

class HealthResponse(BaseModel):
    status: str

class CaptionResponse(BaseModel):
    success: bool
    caption: str
