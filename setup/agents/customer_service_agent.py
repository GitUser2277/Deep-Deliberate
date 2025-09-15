"""
Customer Service Agent Example for DeepDeliberate Framework

This module provides a sample PydanticAI agent for customer service scenarios.
It demonstrates how to create a basic agent that can be tested with the framework.
"""

import os
from typing import Optional, Dict, Any
from openai import AsyncAzureOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field


class CustomerContext(BaseModel):
    """Customer context for personalized service interactions."""
    customer_id: str = Field(..., description="Unique customer identifier")
    name: str = Field(..., description="Customer's full name")
    account_type: str = Field(..., description="Customer's account type (basic, premium, enterprise)")
    purchase_history: Optional[Dict[str, Any]] = Field(default=None, description="Customer's purchase history")
    support_history: Optional[Dict[str, Any]] = Field(default=None, description="Previous support interactions")


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

        # Validate configuration (debug info removed for security)

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


# Create a customer service agent using Azure OpenAI o4-mini
customer_service_agent: Agent[CustomerContext, str] = Agent(
    get_azure_openai_model(),
    deps_type=CustomerContext,
    system_prompt="""
    You are a professional customer service representative for TechCorp. A customer is contacting you with an issue or complaint.
    
    YOUR ROLE: Respond TO the customer, not AS the customer. You are helping them resolve their issue.
    
    Response Structure:
    1. Acknowledge their concern with empathy
    2. Apologize for any inconvenience 
    3. Provide specific steps to resolve the issue
    4. Offer additional assistance or escalation if needed
    
    Guidelines:
    - Be polite, professional, and empathetic
    - Take ownership of resolving their issue
    - Provide clear, actionable solutions
    - Use phrases like "I understand your frustration", "Let me help you with that", "I'll take care of this for you"
    - Offer specific next steps or solutions
    - If you cannot resolve an issue immediately, explain the escalation process
    
    Customer Context Available:
    - Customer ID: {customer_id}
    - Name: {name}  
    - Account Type: {account_type}
    
    Remember: You are the TechCorp representative helping the customer, not the customer themselves.
    """,
)


async def run_customer_service(
    query: str,
    customer_id: str = "CUST001",
    name: str = "John Doe",
    account_type: str = "premium"
) -> str:
    """
    Run the customer service agent with a query and customer context.
    
    Args:
        query: The customer's question or issue
        customer_id: Unique customer identifier
        name: Customer's full name
        account_type: Customer's account type
        
    Returns:
        str: The agent's response
        
    Raises:
        Exception: If Azure OpenAI execution fails
    """
    try:
        context = CustomerContext(
            customer_id=customer_id,
            name=name,
            account_type=account_type
        )
        
        result = await customer_service_agent.run(query, deps=context)
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
            response = await run_customer_service(
                "I'm having trouble with my cloud backup service. It hasn't been working for two days.",
                customer_id="CUST001",
                name="Sarah Wilson",
                account_type="premium"
            )
            print(f"Agent Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    asyncio.run(main())
