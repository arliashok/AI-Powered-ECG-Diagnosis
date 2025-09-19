"""
Minimal LLM Client Interface using get_azure_openai_direct_response()
Only supports messages parameter for querying Azure OpenAI
"""

import logging
from typing import List, Dict
from dataclasses import dataclass


from idam_token_generator import IDAMTokenGenerator
import LLM_config as config
from openai import AzureOpenAI


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMMessage:
    role: str  # "system", "user", "assistant"
    content: str

@dataclass
class LLMResponse:
    content: str

def get_azure_openai_direct_response(messages: List[Dict[str, str]]) -> str:
    """
    Query Azure OpenAI directly using stored config and IDAM token generator
    """
    idam = IDAMTokenGenerator(
        config.IDAM_TOKEN_ENDPOINT,
        config.IDAM_APP_CLIENT_ID,
        config.IDAM_APP_CLIENT_SECRET,
        config.IDAM_LMAAS_APP_AUDIENCE
    )
    llm = AzureOpenAI(
        azure_endpoint=config.OPENAI_ENDPOINT,
        azure_deployment=config.OPENAI_DEPLOYMENT_MODEL,
        api_version=config.OPENAI_AZURE_API_VERSION,
        azure_ad_token=idam.get_idam_token()
    )
    response = llm.chat.completions.create(
        model=config.OPENAI_DEPLOYMENT_MODEL,
        temperature=0.0,
        messages=messages,
        top_p=1.0
    )
    logger.info(f"Azure OpenAI response received")
    print(response.choices[0].message.content)
    return response.choices[0].message.content

async def generate_response(messages: List[LLMMessage]) -> LLMResponse:
    """
    Async wrapper to query Azure OpenAI.
    Only accepts a list of LLMMessage instances as input.
    """
    try:
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
        print(message_dicts)

        import asyncio
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, get_azure_openai_direct_response, message_dicts)

        return LLMResponse(content=content)

    except Exception as e:
        logger.error(f"Error querying Azure OpenAI: {e}")
        raise
