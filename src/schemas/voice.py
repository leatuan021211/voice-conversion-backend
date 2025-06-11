from typing import Optional
from pydantic import BaseModel, Field
import torch

class Voice(BaseModel):
    """
    Represents a voice with its name and feature vector.
    
    Attributes:
        name (str): The name of the voice.
        feature_vector (list): The feature vector for the voice.
    """
    name: str = Field(..., description="The name of the voice")
    description: str = Field(..., description="A brief description of the voice")
    feature_vector: Optional[torch.Tensor] = Field(..., description="The feature vector for the voice")
    feature_vector_path: str = Field(..., description="Path to the feature vector file")
    is_active: bool = Field(True, description="Indicates if the voice is active")
    
    def load_feature_vector(self):
        """
        Loads the feature vector from the specified path.
        
        Returns:
            torch.Tensor: The loaded feature vector.
        """
        self.feature_vector = torch.load(self.feature_vector_path)
        return self.feature_vector
