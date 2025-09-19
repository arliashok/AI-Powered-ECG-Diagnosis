import requests

MCP_BASE_URL = "http://localhost:5000"

def discover_tools():
    response = requests.get(f"{MCP_BASE_URL}/.well-known/mcp.json")
    response.raise_for_status()
    return response.json().get("tools", [])

def invoke_tool(tool_name, payload):
    url = f"{MCP_BASE_URL}/invoke/{tool_name}"
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()
