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
            logger.info(f"Added message to history: {role} - {content[:100]}...")

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
            prompt_result = await self.session.get_prompt(name, arguments)
            return prompt_result
        except Exception as e:
            error_msg = f"Error getting prompt {name}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
    # ============================================================
    # Process a Query using llm Model and Available Tools
    # ============================================================
    async def process_query(self, query: str) -> str:
        """Process a query using Qwen model and available tools
        Args:
            query: The query to process
        Result:
            The response from the AI after processing the query
        """
        #Add user query to history
        await self.add_to_history("user", query)
        messages = []   
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })

        #We need to properly maintain the tool call sequence
        #This means ensuring every 'tool' message follows an 'assistant' message with tool_calls
        assistant_with_tool_calls = None
        pending_tool_responses = []

        #Track message indices to help with debugging
        for i, msg in enumerate(self.message_history):
            #Handle different message types
            if msg['role'] == 'user':
                #first flush any pending tool responses if needed
                if assistant_with_tool_calls and pending_tool_responses:
                    messages.append(assistant_with_tool_calls)
                    messages.extend(pending_tool_responses)
                    assistant_with_tool_calls = None
                    pending_tool_responses = []

                #Then add the user message
                messages.append({
                    "role": "user",
                    "content": msg['content']
                })
            elif msg['role'] == "assistant":
                #check if this is an assistant message with tool calls
                metadata = msg.get('metadata', {})
                if metadata.get('has_tool_calls', False):
                    #If we already have a pending assistant with tool calls, flush it
                    if assistant_with_tool_calls:
                        messages.append(assistant_with_tool_calls)
                        messages.extend(pending_tool_responses)
                        pending_tool_responses = []
                    
                    #Store this assistant message for later (Unitl we collect all tool responses)
                    assistant_with_tool_calls = {
                        "role": "assistant",
                        "content": msg['content'],
                        "tool_calls": metadata.get('tool_calls', [])
                    }
                else:
                    #Regular assistant message without tool calls
                    #first flush any pending tool calls
                    if assistant_with_tool_calls:
                        messages.append(assistant_with_tool_calls)
                        messages.extend(pending_tool_responses)
                        assistant_with_tool_calls = None
                        pending_tool_responses = []
                    
                    #Then add the regular assistant message
                    messages.append({
                        "role": "assistant",
                        "content": msg['content']
                    })

            elif msg['role'] == 'system':
                #System messages can be added directly
                messages.append({
                    "role": "system",
                    "content": msg['content']
                })
            
            elif msg['role'] == 'tool' and "tool_call_id" in msg.get("metadata", {}):
                #Collect tool responses
                if assistant_with_tool_calls:
                    pending_tool_responses.append({
                        "role": "tool",
                        "tool_call_id" : msg['metadata']['tool_call_id'],
                        "content": msg['content']
                    })
            
        #Flush any remaining pending tool calls in at the end
        if assistant_with_tool_calls:
            messages.append(assistant_with_tool_calls)
            messages.extend(pending_tool_responses)

        if self.debug:
            logger.info(f"Prepared {len(messages)} messages for LLM (Qwen3)")
            for i, msg in enumerate(messages):
                role = msg['role']
                has_tool_calls = 'tool_calls' in msg
                preview = msg['content'][:50] + "..." if msg['content'] else ""
                logger.info(f"Message {i}: {role} {'with tool_calls' if has_tool_calls else "(No tool_calls)"} - {preview}")
        
        #Make sure we have the latest tools
        if not self.available_tools:
            await self.refresh_capabilities()

        #Format tools for llm (Qwen3)
        available_tools = [{
            "type": "function",
            "function": {
                'name': tool.name,
                'description': tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in self.available_tools ]

        if self.debug:
            tool_names = [tool['function']['name'] for tool in available_tools]
            logger.info(f"Available tools for query: {tool_names}")
            logger.info(f"Sending {len(messages)} messages to Qwen3")

        #Initial LLM Call
        try: #Sort this out into Qwen3 format
            response = self.model.OllamaChat.create(
                model="qwen3-8b",
                messages=messages,
                tools=available_tools,
                tool_choice="auto"
            )
        except Exception as e:
            error_msg = f"Error calling Qwen Model: {str{e}}"
            logger.error(error_msg)
            await self.add_to_history("assistant", error_msg, {"error": True})
            return error_msg
        
        #Process response and handle tool calls
        tool_results = []
        final_text = []

        assistant_message = response.choice[0].message
        initial_response = assistant_message or ""

        #Add initial assistant response to history with metadata about tool calls
        tool_calls_metadata = {}
        if assistant_message.tool_calls:
            tool_calls_metadata = {
                "has_tool_calls": True,
                "tool_calls": assistant_message.tool_calls
            }

        await self.add_to_history("assistant", initial_response, tool_calls_metadata)
        final_text.append(initial_response)

        #Check if tool calls are present
        if assistant_message.tool_calls:
            if self.debug:
                logger.info(f"Tool calls requested: {len(assistant_message.tool_calls)} tool calls from the llm")

        #Add the assistant's message to the conversation
        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": assistant_message.tool_calls
            }
        )

        #Process each tool call
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments

            #convert json string to dict if needed
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except Exception as e:
                    logger.warning(f"Failed to parse tool argument as JSON: {tool_args}")
                    tool_args = {}

            if self.debug:
                logger.info(f"Executing tool: {tool_name}")
                logger.info(f"Arguments: {tool_args}")

            #Execute tool call on the server
            try:
                result = await self.session.call_tool(tool_name, tool_args)
                tool_content = result.content if hasattr(result, 'content') else str(result)
                tool_results.append({"call": tool_name, "result": tool_content[0].text})
                final_text.append(f"\n[Calling tool {tool_name} with args {tool_args}]")

                if self.debug:
                    result_preview = tool_content[0].text[:100] + "..." if len(tool_content[0].text) > 100 else tool_content[0].text
                    logger.info(f"Tool result preview: {result_preview}")
                
                #Add the tool result to the conversation
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content[0].text
                    }
                )
                await self.add_to_history("tool", tool_content[0].text,
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "tool_call_id": tool_call.id
                    })
            except Exception as e:
                error_msg = f"Error executing tool {tool_name}: {str(e)}"
                logger.error(error_msg)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": error_msg
                })
                await self.add_to_history("tool", error_msg, {"tool": tool_name, "error": True, "tool_call_id": tool_call.id})
                final_text.append(f"\n[Error executing tool {tool_name}:{str(e)}]")
        
        if self.debug:
            logger.info("Getting final response from llm(Qwen3) with tool results")

        #Get a new response from the llm with tool results
        try:
            second_response = self.model.OllamaChat.create(
                model="Qwen3-8b",
                messages=messages
            )

            response_content = second_response.choices[0].message.content or ""
            await self.add_to_history("assistant", response_content)
            final_text.append("\n" + response_content)
        except Exception as e:
            
