import asyncio
import json
import logging
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_ollama import OllamaLLM, ChatOllama


# ============================================================
# Logging
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("document-search-mcp")

#Disable httpx loggers
logging.getLogger("httpx").setLevel(logging.WARNING)

# ============================================================
# MCP Client
# ============================================================

class MCPClient:
    def __init__(self, debug=False):
        """Initialize the MCP Client.
        
        Args:
            debug: Whether to enable logging
        """
        #Initialize session and client obejcts
        self.sessioin: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.debug = debug
        self.message_history = []
        self.system_prompt = "You are a helpful RAG AI assistant named 'RAG-AI-MCP' that can answer questions about the provided documents or query the attached database for more information."
        self.model = OllamaLLM(name="qwen3-8b", temperature=0.7)

        # ============================================================
        # Server Connection info
        # ============================================================

        self.available_tools = []
        self.available_resources = []
        self.available_prompts = []
        self.server_name = None

    # ============================================================
    # Connect to MCP Server
    # ============================================================
    
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
         
         Args:
            server_script_path: Path to the server script (.py or .js)
        """
        if self.debug:
            logger.info(f"Connecting to server at {server_script_path}")

        is_python = server_script_path.endswith(".py")
        if not (is_python):
            raise ValueError("Server script must be a .py file")

        #Initialize server parameters
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None # I think i have env and what i am supposed to put in the dict[str, str]
        )

        #Initialize stdio transport
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

            #initialize the session
            init_result = await self.session.initialize()
            self.server_name = init_result.serverInfo.name

            if self.debug:
                logger.info(f"Connected to server: {self.server_name} v{init_result.serverInfo.version}")
            
            #Cache available tools, resources and prompts
            await self.refresh_capabilities()

            return True
        except Exception as e:
            logging.error(f"Failed to connect to servr: {e}")
            return False
    
    # ============================================================
    # Refresh Server Capabilities
    # ============================================================
    async def refresh_capabilities(self):
        """Refresh the client's knowledge of server capabilities"""
        if not self.session:
            raise ValueError("Not connected to the server")
        
        #Get Available tools
        tools_response = await self.session.list_tools()
        self.available_tools = tools_response.tools

        #Get available resources
        resources_response = await self.session.list_resources()
        self.available_resources = resources_response.resources

        #Get available Prompts
        prompts_response = await self.session.list_prompts()
        self.available_prompts = prompts_response.prompts

        if self.debug:
            logger.info(f"Server capabilities refreshed:")
            logger.info(f"- Tools: {len(self.available_tools)}")
            logger.info(f"- Resources: {len(self.available_resources)}")
            logger.info(f"- Prompts: {len(self.available_prompts)}")

    # ============================================================
    # Handling Message History Helper Function
    # ============================================================
    async def add_to_history(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to the history
        
        Args:
            role: The role of the message sender (user, assistant, system, resource)
            content: The message content
            metadata: Optional metadata about the message
        """
        #Format Message
        message={
            "role": role,
            "content": content,
            "timestamp": asyncio.get_event_loop().time(), #WHY IS THIS KIND OF TIME USED HERE?
            "metadat": metadata or {}
        }
        self.message_history.append(message)

        if self.debug:
            logger.info(f"Added message to historu: {role} - {content[:100]}...")

    # ============================================================
    # List Available Resources from the MCP Server
    # ============================================================
    async def list_resources(self):
        """List available resources from the MCP Server"""
        if not self.session:
            raise ValueError("Not Connnected to the server")   
        
        response = await self.session.list_resources()
        self.available_resources = response.resources

        if self.debug:
            resource_uris = [res.uri for res in self.available_resources]
            logger.info(f"Available resources: {resource_uris}")

        return self.available_resources
    
    # ============================================================
    # Read Content from a resource and add to Message History
    # ============================================================
    async def read_resource(self, uri: str):
        """Read content from a specific resource
        
        Args:
            uri: The URI of the resource to read
        Returns:
            The content of the resource as a string
        """

        if self.debug:
            logger.info(f"Reading resource: {uri}")
        try:
            if not self.session:
                raise ValueError("Not connected to the server")

            result = await self.session.read_resource(uri)
            if not result:
                content = "No content fount for this resource"
            else:
                content = result if isinstance(result, str) else str(result)

            #Add resource content to history as a user message
            resource_message = f"Resource content from {uri}:\n\n{content}"
            await self.add_to_history("user", resource_message, {"resource_uri": uri, "is_resource": True })

            return content
        except Exception as e:
            error_msg = f"Error reading resource {uri}: {str(e)}"
            logger.error(error_msg)
            await self.add_to_history("user", error_msg, {"uri": uri, "error": True})
            return error_msg
        
    # ============================================================
    # List Available Prompts from the MCP Server
    # ============================================================
    async def list_prompts(self):
        """List available prompts for the server"""
        if not self.session:
            raise ValueError("Not Connected to the Server")
        
        response = await self.session.list_prompts()
        self.available_prompts = response.prompts

        if self.debug:
            prompt_names = [prompt.name for prompt in self.available_prompts]
            logger.info(f"Available Prompts: {prompt_names}")

        return self.available_prompts
    
    # ============================================================
    # Get a Specific Prompt with Arguments
    # ============================================================
    async def get_prompt(self, name: str, arguments: dict = None):
        """Get a specific prompt with arguments
        Args:
            name: The name of the prompt
            arguments: Optional arguments to pass to the prompt
        Returns:
            The Prompt result
        """

        if self.debug:
            logger.info(f"Getting Prompt: {name} with arguments: {arguments}")

        try:
            prompt_result = await self.session.get_prompt(name, argurments)
            return prompt_result
        except Exception as e:
            error_msg = f"Error getting prompt {name}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
    # ============================================================
    # Process a Query using llm Model and Available Tools
    # ============================================================