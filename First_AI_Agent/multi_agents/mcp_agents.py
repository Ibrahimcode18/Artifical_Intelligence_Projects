import asyncio
import base64
from dotenv import load_dotenv
from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

async def main():
    # 1. Load Environment
    load_dotenv()
    print("✅ Environment variables loaded.")

    retry_config = types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504],
    )

    # Setup MCP Server
    print("Starting MCP Server...")
    mcp_image_server = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="npx.cmd",
                args=["-y", "@modelcontextprotocol/server-everything"],
                tool_filter=["getTinyImage"],
            ),
            timeout=120, # 120 seconds to allow for Node startup
        )
    )
    print("MCP Tool Created.")

    # Add MCP Tool to Agent
    image_agent = LlmAgent(
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        name="image_agent",
        instruction="Use the MCP Tool to generate images for user queries",
        tools=[mcp_image_server],
    )
    
    # Create Runner and Execute
    runner = InMemoryRunner(agent=image_agent)
    print("Agent is thinking...")
    response = await runner.run_debug("Provide a sample tiny image", verbose=True)

    # 5. Process Output
    for event in response:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_response") and part.function_response:
                    for item in part.function_response.response.get("content", []):
                        if item.get("type") == "image":
                            # Save the image to disk
                            img_data = base64.b64decode(item["data"])
                            with open("tiny_image_output.png", "wb") as f:
                                f.write(img_data)
                            print("Image successfully saved to 'tiny_image_output.png'!")

if __name__ == "__main__":
    asyncio.run(main())