import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from llm_client import LLMMessage, generate_response
from action_executor import ActionExecutor
import mcp_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentMemory:
    short_term: Dict[str, Any] = field(default_factory=dict)
    long_term: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class AgentContext:
    goal: str = ""
    current_task: str = ""
    available_tools: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class AgenticAISystem:
    def __init__(self):
        self.memory = AgentMemory()
        self.context = AgentContext()
        self.session_id = str(uuid.uuid4())
        self.executor = ActionExecutor()
        self._setup_tools()

    def _setup_tools(self):
        try:
            tools = mcp_client.discover_tools()
            self.context.available_tools = [tool["name"] for tool in tools]
            logger.info(f"Discovered MCP tools: {self.context.available_tools}")
        except Exception as e:
            logger.error(f"Failed to discover MCP tools: {e}")
            self.context.available_tools = []

    async def process_query(self, query: str) -> Dict[str, Any]:
        logger.info(f"Processing query: {query}")
        self.context.goal = query
        self.context.current_task = query
        self.memory.conversation_history.append({
            "role": "user",
            "content": query,
            "timestamp": datetime.now().isoformat()
        })

        system_prompt = self._build_system_prompt()
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=query)
        ]

        llm_resp = await generate_response(messages)
        llm_content = llm_resp.content
        logger.info(f"LLM response: {llm_content}")

        # Parse and execute plan steps via ActionExecutor
        result = await self.executor.parse_and_execute(llm_content, self.context.available_tools)

        self.memory.conversation_history.append({
            "role": "assistant",
            "content": result["response"],
            "timestamp": datetime.now().isoformat()
        })

        return {
            "response": result["response"],
            "tool_results": result.get("tool_results"),
            "session_id": self.session_id
        }

    def _build_system_prompt(self) -> str:
        tools_str = ", ".join(self.context.available_tools) if self.context.available_tools else "get_patient_data, run_diagnostic_test"
        return (
            f"You can use these MCP tools: {tools_str}.\n"
            "Respond with a JSON list of steps, each a dict with 'tool' and 'parameters'.\n"
            "Example:\n"
            '[\n'
            '  {"tool": "get_patient_data", "parameters": {"patient_id": "12345"}},\n'
            '  {"tool": "run_diagnostic_test", "parameters": {"patient_id": "12345"}}\n'
            ']\n'
        )

    async def close(self):
        pass
