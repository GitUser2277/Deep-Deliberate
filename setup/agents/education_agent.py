"""
Education Agent Example for DeepDeliberate Framework

This module provides a sample PydanticAI agent for educational scenarios.
It demonstrates how to create a tutor agent that can be tested with the framework.
"""

import os
from typing import Optional, List, Dict
from openai import AsyncAzureOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field


class StudentContext(BaseModel):
    """Student context for personalized educational interactions."""
    student_id: str = Field(..., description="Unique student identifier")
    name: str = Field(..., description="Student's full name")
    grade_level: str = Field(..., description="Student's grade level or education stage")
    subjects: List[str] = Field(default_factory=list, description="Subjects the student is studying")
    learning_style: Optional[str] = Field(default=None, description="Student's preferred learning style")
    previous_questions: Optional[List[Dict[str, str]]] = Field(default=None, description="Previous questions asked")


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


# Create an education agent using Azure OpenAI o4-mini
education_agent: Agent[StudentContext, str] = Agent(
    get_azure_openai_model(),
    deps_type=StudentContext,
    system_prompt="""
    You are an educational tutor specialized in helping students learn effectively.
    
    Guidelines:
    - Adapt your teaching style to the student's grade level and learning preferences
    - Use clear, age-appropriate language and examples
    - Encourage critical thinking and problem-solving
    - Provide step-by-step explanations for complex topics
    - Use positive reinforcement and constructive feedback
    - Connect new concepts to the student's existing knowledge
    - Suggest additional resources and practice opportunities
    
    Available student context:
    - Student ID and name
    - Grade level or education stage
    - Subjects being studied
    - Learning style preferences (visual, auditory, kinesthetic, etc.)
    - Previous questions and topics covered
    
    Remember to be patient, encouraging, and supportive. Your goal is to help 
    students understand concepts deeply and develop independent learning skills.
    """,
)


async def run_education_session(
    question: str,
    student_id: str = "STU001",
    name: str = "Alex Smith",
    grade_level: str = "8th grade",
    subjects: List[str] = None,
    learning_style: str = "visual"
) -> str:
    """
    Run the education agent with a question and student context.
    
    Args:
        question: The student's question or topic to learn about
        student_id: Unique student identifier
        name: Student's full name
        grade_level: Student's grade level or education stage
        subjects: List of subjects the student is studying
        learning_style: Student's preferred learning style
        
    Returns:
        str: The tutor's response
        
    Raises:
        Exception: If Azure OpenAI execution fails
    """
    try:
        if subjects is None:
            subjects = ["mathematics", "science", "english"]
        
        context = StudentContext(
            student_id=student_id,
            name=name,
            grade_level=grade_level,
            subjects=subjects,
            learning_style=learning_style
        )
        
        result = await education_agent.run(question, deps=context)
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
            response = await run_education_session(
                "Can you explain how photosynthesis works? I'm having trouble understanding the process.",
                student_id="STU001",
                name="Emma Johnson",
                grade_level="9th grade",
                subjects=["biology", "chemistry", "mathematics"],
                learning_style="visual"
            )
            print(f"Tutor Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    asyncio.run(main())
