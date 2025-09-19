# Agentic AI System with Model Context Protocol (MCP)

A complete implementation of an autonomous AI agent system that uses cloud LLM APIs to query for action plans and executes them using the Model Context Protocol (MCP) for client-server communication.

## 🎯 Features

- **Cloud LLM Integration**: Support for OpenAI, Anthropic, and other LLM providers
- **Model Context Protocol**: Standardized communication between AI agents and tools
- **Action Plan Parsing**: Parse LLM responses into structured, executable action plans
- **Autonomous Execution**: Execute multi-step plans with tools and APIs
- **Memory & Context**: Maintain conversation history and execution context
- **Tool Integration**: Built-in tools for calculations, web search, system commands, and file operations
- **Reflection & Learning**: Agent can reflect on results and adapt behavior
- **Interactive Interface**: User-friendly command-line interface

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Agentic AI     │───▶│   LLM Client    │
└─────────────────┘    │     System      │    │  (OpenAI/       │
                       └─────────────────┘    │   Anthropic)    │
                                │             └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Action Parser   │───▶│ Action Executor │
                       │   & Planner     │    │                 │
                       └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  MCP Client     │◀───│     Tools &     │
                       │                 │    │   Resources     │
                       └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   MCP Server    │
                       │  (Tools, APIs,  │
                       │   Resources)    │
                       └─────────────────┘
```

## 📁 Project Structure

```
agentic-ai-system/
├── config.py                 # Configuration management
├── llm_client.py             # Cloud LLM API integration
├── mcp_protocol.py           # Model Context Protocol implementation
├── action_executor.py        # Action parsing and execution
├── agentic_ai_system.py      # Main orchestrator
├── example_server.py         # Example MCP server
├── main_client.py            # Interactive client application
├── requirements.txt          # Python dependencies
├── .env.template            # Environment variables template
├── setup.sh                 # Setup script
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or download the code files
# Navigate to the project directory

# Run setup script (Linux/Mac)
chmod +x setup.sh
./setup.sh

# Or manual setup:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy environment template
cp .env.template .env

# Edit .env file with your API keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### 3. Run the System

```bash
# Interactive mode
python main_client.py

# Demo mode
python main_client.py --demo
```

## 💡 Usage Examples

### Basic Queries
```
👤 You: Calculate the compound interest for $1000 at 5% for 3 years

🤖 Assistant: I'll calculate the compound interest for you.

📋 EXECUTION SUMMARY:
- Plan Status: COMPLETED
- Actions Executed: 1

🔧 ACTIONS PERFORMED:
   1. ✅ calculate - completed
      Result: Compound Interest: $157.63, Final Amount: $1157.63
```

### Complex Multi-Step Tasks
```
👤 You: Research the latest AI developments and create a summary report

🤖 Assistant: I'll research AI developments and create a comprehensive summary.

📋 EXECUTION SUMMARY:
- Plan Status: COMPLETED  
- Actions Executed: 3

🔧 ACTIONS PERFORMED:
   1. ✅ web_search - completed
   2. ✅ analyze_data - completed  
   3. ✅ write_file - completed
      Result: Summary report saved to ai_developments_2025.md
```

## 🔧 Core Components

### 1. LLM Client (`llm_client.py`)
- Unified interface for multiple LLM providers
- Support for tool calling and function execution
- Async/await pattern for concurrent operations
- Error handling and retry logic

### 2. MCP Protocol (`mcp_protocol.py`)
- JSON-RPC 2.0 message format
- STDIO and HTTP transport protocols
- Client-server capability negotiation
- Tool and resource management

### 3. Action System (`action_executor.py`)
- Parse LLM responses into structured action plans
- Support for multiple action types:
  - Tool calls
  - API requests  
  - Data queries
  - Workflows
  - Decisions
  - Reflections
  - Memory updates

### 4. Agentic AI System (`agentic_ai_system.py`)
- Main orchestrator for autonomous behavior
- Memory and context management
- Multi-step plan execution
- Reflection and learning capabilities

## 🛠️ Available Tools

### Built-in Tools
- **calculate**: Mathematical expressions and calculations
- **execute_command**: System command execution
- **web_search**: Web search and information retrieval  
- **analyze_data**: Statistical data analysis
- **read_file**: File reading operations
- **write_file**: File writing operations

### MCP Server Tools
The example MCP server provides additional tools that can be extended:
- Calculator with advanced functions
- System information queries
- Data processing capabilities
- Resource management

## 🔌 Extending the System

### Adding New Tools

1. **Register in Action Executor**:
```python
async def my_custom_tool(param1: str, param2: int) -> str:
    # Your tool implementation
    return f"Processed {param1} with {param2}"

executor.register_tool("my_tool", my_custom_tool)
```

2. **Add to MCP Server**:
```python
server.add_tool(
    name="my_tool",
    description="Description of what the tool does",
    schema={
        "type": "object",
        "properties": {
            "param1": {"type": "string"},
            "param2": {"type": "integer"}
        },
        "required": ["param1", "param2"]
    },
    handler=my_custom_tool
)
```

### Adding New LLM Providers

Extend the `LLMClient` class in `llm_client.py`:

```python
async def _custom_provider_generate(self, messages, tools, **kwargs):
    # Implementation for your LLM provider
    # Return LLMResponse object
    pass
```

## 🔒 Security Considerations

- **API Key Protection**: Store API keys in environment variables
- **Command Execution**: Validate and sanitize system commands
- **Tool Access**: Implement proper authorization for tool usage
- **Data Privacy**: Handle sensitive data according to privacy requirements
- **Error Handling**: Graceful degradation and error recovery

## 📊 Configuration Options

### Environment Variables
```bash
# LLM Settings
LLM_MODEL=gpt-4                    # Model to use
LLM_TEMPERATURE=0.7               # Creativity level (0-1)
LLM_MAX_TOKENS=2000              # Maximum response length

# Agent Behavior  
AGENT_MAX_ITERATIONS=10          # Maximum planning iterations
AGENT_REFLECTION=true            # Enable reflection capabilities
AGENT_MEMORY=true               # Enable memory systems

# MCP Configuration
MCP_TRANSPORT=stdio              # Transport protocol (stdio/http)
MCP_TIMEOUT=30                  # Connection timeout
```

## 🧪 Testing

### Unit Tests
```bash
# Run basic functionality tests
python -m pytest tests/ -v
```

### Integration Tests
```bash
# Test with MCP server
python test_integration.py
```

### Demo Scenarios
```bash
# Run predefined demo scenarios
python main_client.py --demo
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Not Set**:
   - Ensure `.env` file has valid API keys
   - Check environment variable loading

2. **MCP Connection Failed**:
   - Verify server script path
   - Check Python executable in PATH
   - Review server logs for errors

3. **Tool Execution Timeout**:
   - Increase `TOOL_TIMEOUT` setting
   - Check network connectivity
   - Verify tool parameters

4. **Memory Issues**:
   - Monitor conversation history size
   - Implement history pruning
   - Use efficient data structures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation  
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Model Context Protocol by Anthropic
- OpenAI and Anthropic for LLM APIs
- Python async/await ecosystem
- JSON-RPC specification

## 📚 Additional Resources

- [Model Context Protocol Specification](https://modelcontextprotocol.io)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com)
- [Agentic AI Research Papers](https://arxiv.org/search/?query=agentic+AI)

---

**Built with ❤️ for autonomous AI agents**
