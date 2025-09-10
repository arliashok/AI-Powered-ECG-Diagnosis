"""
Test Connection Script for LMaaS
Demonstrates different ways to connect to and use LMaaS services
"""
from openai import AzureOpenAI
import boto3
from langchain.chat_models import AzureChatOpenAI, BedrockChat
from langchain.schema import HumanMessage

import config
from idam_token_generator import IDAMTokenGenerator

def test_azure_openai_direct():
    """Test Azure OpenAI connection using direct SDK"""
    print("\n=== Testing Azure OpenAI (Direct SDK) ===")

    # Initialize IDAM token generator
    idam = IDAMTokenGenerator(
        config.IDAM_TOKEN_ENDPOINT,
        config.IDAM_APP_CLIENT_ID,
        config.IDAM_APP_CLIENT_SECRET,
        config.IDAM_LMAAS_APP_AUDIENCE
    )

    # Initialize Azure OpenAI client
    llm = AzureOpenAI(
        azure_endpoint=config.OPENAI_ENDPOINT,
        azure_deployment=config.OPENAI_DEPLOYMENT_MODEL,
        api_version=config.OPENAI_AZURE_API_VERSION,
        azure_ad_token=idam.get_idam_token()
    )

    # Test query
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is artificial intelligence?"},
    ]

    response = llm.chat.completions.create(
        model=config.OPENAI_DEPLOYMENT_MODEL,
        temperature=0.0,
        messages=messages,
        top_p=1.0
    )

    print("Response:", response.choices[0].message.content)

def test_azure_openai_langchain():
    """Test Azure OpenAI connection using LangChain"""
    print("\n=== Testing Azure OpenAI (LangChain) ===")

    # Initialize IDAM token generator
    idam = IDAMTokenGenerator(
        config.IDAM_TOKEN_ENDPOINT,
        config.IDAM_APP_CLIENT_ID,
        config.IDAM_APP_CLIENT_SECRET,
        config.IDAM_LMAAS_APP_AUDIENCE
    )

    # Initialize LangChain Azure OpenAI chat model
    llm = AzureChatOpenAI(
        openai_api_base=config.OPENAI_ENDPOINT,
        openai_api_version=config.OPENAI_AZURE_API_VERSION,
        deployment_name=config.OPENAI_DEPLOYMENT_MODEL,
        openai_api_key="unused",
        azure_ad_token=idam.get_idam_token()
    )

    # Test query
    response = llm.invoke([HumanMessage(content="What is machine learning?")])
    print("Response:", response.content)

def test_bedrock_direct():
    """Test AWS Bedrock connection using direct SDK"""
    print("\n=== Testing AWS Bedrock (Direct SDK) ===")

    # Initialize IDAM token generator
    idam = IDAMTokenGenerator(
        config.IDAM_TOKEN_ENDPOINT,
        config.IDAM_APP_CLIENT_ID,
        config.IDAM_APP_CLIENT_SECRET,
        config.IDAM_LMAAS_APP_AUDIENCE
    )

    # Initialize Bedrock client
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=config.BEDROCK_REGION,
        endpoint_url=config.BEDROCK_ENDPOINT,
        aws_access_key_id="unused",
        aws_secret_access_key="unused",
        aws_session_token=idam.get_idam_token()
    )

    # Test query with Claude model
    prompt = "Human: What is deep learning?\nAssistant:"
    response = bedrock.invoke_model(
        modelId=config.BEDROCK_MODEL_ID,
        body={"prompt": prompt}
    )
    print("Response:", response['body'].read().decode())

def main():
    """Main function to run all tests"""
    try:
        # Test Azure OpenAI implementations
        test_azure_openai_direct()
        # test_azure_openai_langchain()

        # Test AWS Bedrock implementation
        # test_bedrock_direct()

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
