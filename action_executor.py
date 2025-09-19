import asyncio
import json
import logging
from typing import List, Dict, Any
import mcp_client

logger = logging.getLogger(__name__)

class ActionExecutor:

    async def parse_and_execute(self, llm_content: str, available_tools: List[str]) -> Dict[str, Any]:
        # Parse the LLM generated JSON list of tool steps
        try:
            steps = json.loads(llm_content)
            if not isinstance(steps, list):
                raise ValueError("LLM response JSON is not a list")
        except Exception as e:
            logger.error(f"Failed to parse LLM response JSON: {e}")
            return {"response": llm_content, "tool_results": None}

        tool_results = []
        for step in steps:
            tool_name = step.get("tool")
            parameters = step.get("parameters", {})

            if tool_name not in available_tools:
                logger.warning(f"Requested tool '{tool_name}' not available")
                tool_results.append({"tool": tool_name, "error": "Tool not available"})
                continue

            try:
                logger.info(f"Invoking tool '{tool_name}' with parameters {parameters}")
                # Invoke the tool through MCP HTTP client
                result = mcp_client.invoke_tool(tool_name, parameters)
                tool_results.append({"tool": tool_name, "result": result})
            except Exception as e:
                logger.error(f"Error invoking tool '{tool_name}': {e}")
                tool_results.append({"tool": tool_name, "error": str(e)})

        # Build combined response
        combined_response = f"Executed {len(tool_results)} steps:\n"
        for tr in tool_results:
            if "result" in tr:
                combined_response += f"Tool '{tr['tool']}' result:\n{tr['result']}\n"
            else:
                combined_response += f"Tool '{tr['tool']}' error:\n{tr.get('error', 'Unknown error')}\n"

        return {
            "response": combined_response,
            "tool_results": tool_results
        }
