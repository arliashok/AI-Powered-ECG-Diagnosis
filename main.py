import asyncio
import logging
import platform
import subprocess
import sys
import time
from typing import Dict, Any
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed.")

from agentic_ai_system import AgenticAISystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgenticAIClient:
    def __init__(self):
        self.system = None
        self.session_active = False
        self.is_windows = platform.system() == "Windows"
        self.server_process = None

    def start_mcp_server(self):
        """Start the Flask MCP server as a subprocess."""
        server_script_path = Path(__file__).parent / "mcp_server.py"
        python_executable = sys.executable  # Current Python interpreter

        try:
            # Start server, suppressing stdout/stderr or redirect as needed
            self.server_process = subprocess.Popen(
                [python_executable, str(server_script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info(f"MCP server started with PID: {self.server_process.pid}")

            # Optional: wait a few seconds for the server to initialize
            time.sleep(3)
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise

    async def initialize(self, config: Dict[str, Any] = None):
        logger.info("Initializing Agentic AI Client...")

        # Start MCP server before initializing the system
        self.start_mcp_server()

        self.system = AgenticAISystem()
        await asyncio.sleep(0)  # placeholder for async init if needed
        self.session_active = True
        logger.info("Agentic AI Client initialized!")

    async def run_interactive_session(self):
        if not self.session_active:
            await self.initialize()
        self._print_welcome()
        try:
            while True:
                user_input = input("\n👤 You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🤖 Goodbye! Thanks for using the Agentic AI System.")
                    break
                if user_input.lower() == 'help':
                    self._print_help()
                    continue
                if user_input.lower() == 'status':
                    self._print_status()
                    continue
                if user_input.lower() == 'clear':
                    self._clear_console()
                    continue
                if not user_input:
                    continue

                print("\n🤖 Assistant: Thinking and planning...")
                result = await self.system.process_query(user_input)
                print(f"\n🤖 Assistant: {result['response']}")
                if 'tool_result' in result:
                    print(f"\nTool Result:\n{result['tool_result']}")

        except KeyboardInterrupt:
            print("\n\n🤖 Session interrupted. Goodbye!")
        finally:
            await self.close()

    def _print_welcome(self):
        separator = "=" * 60
        print(f"\n{separator}")
        print("AGENTIC AI SYSTEM - INTERACTIVE SESSION")
        print(separator)
        print("Ask me anything! I can plan and execute complex tasks.")
        print("Available capabilities:\n- run_diagnostic_test\n- get_patient_data")
        print("\nType 'quit' to exit, 'help' for commands\n" + "-"*60)

    def _print_help(self):
        print("\nAVAILABLE COMMANDS:")
        print(" - help      : Show this help message")
        print(" - status    : Show system status")
        print(" - clear     : Clear the screen")
        print(" - quit      : Exit the session")
        print("\nEXAMPLE QUERIES:")
        print(" - Get patient data for patient ID 123")
        print(" - Run diagnostic test for patient ID 123")

    def _print_status(self):
        print("\nSYSTEM STATUS:")
        print(f" - Session ID: {self.system.session_id}")
        print(f" - Available Tools: {', '.join(self.system.context.available_tools)}")
        print(f" - Current Goal: {self.system.context.goal or 'None'}")

    def _clear_console(self):
        import os
        command = 'cls' if self.is_windows else 'clear'
        os.system(command)

    async def close(self):
        if self.system:
            await self.system.close()
        self.session_active = False

        if self.server_process:
            logger.info("Stopping MCP server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
                logger.info("MCP server stopped.")
            except subprocess.TimeoutExpired:
                logger.warning("MCP server did not terminate in time; killing process.")
                self.server_process.kill()
            self.server_process = None

async def main():
    client = AgenticAIClient()
    try:
        await client.run_interactive_session()
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"\nApplication error: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())