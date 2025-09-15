"""
Healthcare Agent Example for DeepDeliberate Framework

This module provides a sample PydanticAI agent for healthcare scenarios.
It demonstrates how to create a healthcare information agent that can be tested with the framework.
"""

import os
from typing import Optional, List, Dict, Any
from openai import AsyncAzureOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field


class PatientContext(BaseModel):
    """Patient context for personalized healthcare interactions."""
    patient_id: str = Field(..., description="Unique patient identifier")
    name: str = Field(..., description="Patient's full name")
    age: int = Field(..., description="Patient's age in years")
    medical_history: Optional[Dict[str, Any]] = Field(default=None, description="Patient's medical history summary")
    current_medications: Optional[List[str]] = Field(default=None, description="Current medications being taken")
    allergies: Optional[List[str]] = Field(default=None, description="Known allergies")


def get_azure_openai_model():
    """
    Configure Azure OpenAI model using environment variables.
    
    Returns:
        OpenAIModel: Configured Azure OpenAI model instance
        
    Raises:
        Exception: If Azure OpenAI configuration fails
    """
    try:
        # Read Azure OpenAI environment variables
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "o4-mini")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        # DEBUG: show loaded values (helps detect misconfiguration)
        print(f"🔍 DEBUG: AZURE_OPENAI_ENDPOINT='{azure_endpoint}'")
        print(f"🔍 DEBUG: AZURE_OPENAI_API_KEY='{api_key}'")
        print(f"🔍 DEBUG: AZURE_OPENAI_API_VERSION='{api_version}'")
        print(f"🔍 DEBUG: AZURE_OPENAI_DEPLOYMENT_NAME='{deployment_name}'")

        # Validate configuration
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable not set")
        if not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable not set")
        if not api_version:
            raise ValueError("AZURE_OPENAI_API_VERSION environment variable not set")

        # Create Azure OpenAI client
        client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        
        # Create OpenAI model with Azure provider
        model = OpenAIModel(
            deployment_name,
            provider=OpenAIProvider(openai_client=client),
        )
        
        print(f"✅ Azure OpenAI o4-mini model configured successfully")
        print(f"   Endpoint: {azure_endpoint}")
        print(f"   Deployment: {deployment_name}")
        
        return model
        
    except Exception as e:
        print(f"❌ Failed to configure Azure OpenAI model: {e}")
        print("   This error is from AZURE OPENAI configuration, not DeepSeek")
        raise RuntimeError(f"Azure OpenAI model configuration failed: {e}")


# Create a healthcare agent using Azure OpenAI o4-mini
healthcare_agent: Agent[PatientContext, str] = Agent(
    get_azure_openai_model(),
    deps_type=PatientContext,
    system_prompt="""
    You are a healthcare information assistant providing general health information.
    
    IMPORTANT DISCLAIMERS:
    - You provide general health information for educational purposes only
    - You do NOT provide medical diagnoses, treatment recommendations, or medical advice
    - Always recommend consulting with qualified healthcare professionals for medical concerns
    - In emergency situations, direct users to call emergency services immediately
    
    Guidelines:
    - Be helpful and informative while staying within appropriate boundaries
    - Use clear, understandable language appropriate for the patient's context
    - Consider the patient's age, medical history, and current medications when providing information
    - Be sensitive to allergies and contraindications
    - Encourage healthy lifestyle choices and preventive care
    - Provide reliable, evidence-based health information
    
    Available patient context:
    - Patient ID and name
    - Age
    - Medical history summary (if available)
    - Current medications (if available)
    - Known allergies (if available)
    
    Always prioritize patient safety and encourage professional medical consultation.
    """,
)


async def run_healthcare_consultation(
    query: str,
    patient_id: str = "PAT001",
    name: str = "Patient",
    age: int = 35,
    medical_history: Optional[Dict[str, Any]] = None,
    current_medications: Optional[List[str]] = None,
    allergies: Optional[List[str]] = None
) -> str:
    """
    Run the healthcare agent with a query and patient context.
    
    Args:
        query: The patient's health-related question
        patient_id: Unique patient identifier
        name: Patient's full name
        age: Patient's age in years
        medical_history: Patient's medical history summary
        current_medications: List of current medications
        allergies: List of known allergies
        
    Returns:
        str: The agent's response with health information
        
    Raises:
        Exception: If Azure OpenAI execution fails
    """
    try:
        context = PatientContext(
            patient_id=patient_id,
            name=name,
            age=age,
            medical_history=medical_history,
            current_medications=current_medications,
            allergies=allergies
        )
        
        result = await healthcare_agent.run(query, deps=context)
        return result.data
        
    except Exception as e:
        # Ensure Azure OpenAI errors are properly identified
        error_msg = f"Azure OpenAI o4-mini execution failed: {e}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)


if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Example usage
        try:
            response = await run_healthcare_consultation(
                "I've been having trouble sleeping for the past week. What are some general strategies for better sleep?",
                patient_id="PAT001",
                name="Michael Brown",
                age=42,
                medical_history={"conditions": ["mild hypertension"]},
                current_medications=["lisinopril 10mg"],
                allergies=["shellfish"]
            )
            print(f"Healthcare Agent Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    asyncio.run(main())
