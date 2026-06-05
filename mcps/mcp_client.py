import asyncio
import json
import logging
import sys
from typing import Optional, Dict, Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage


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
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.debug = debug
        self.message_history = []
        self.system_prompt = "You are a helpful RAG AI assistant named 'RAG-AI-MCP' that can answer questions about the provided documents or query the attached database for more information."
        self.model = ChatOllama(model="qwen3:8b", temperature=0.7, reasoning=True)

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
    async def add_to_history(self, role: str, content: Any, metadata: Dict[str, Any] = None):
        """Add a message to the history
        
        Args:
            role: The role of the message sender (user, assistant, system, resource)
            content: The message content
            metadata: Optional metadata about the message
        """
        if hasattr(content, 'content'):
            content = content.content if hasattr(content.content, 'text') else str(content.content)
        #Format Message
        message={
            "role": role,
            "content": str(content),
            "timestamp": asyncio.get_event_loop().time(), #WHY IS THIS KIND OF TIME USED HERE?
            "metadata": metadata or {}
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
                content = "No content found for this resource"
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
        #This means ensuring every 'tool' message follows an 'assistant(AI)' message with tool_calls
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
                #check if this is an ai message with tool calls
                metadata = msg.get('metadata', {})
                if metadata.get('has_tool_calls', False):
                    #If we already have a pending assistant with tool calls, flush it
                    if assistant_with_tool_calls:
                        messages.append(assistant_with_tool_calls)
                        messages.extend(pending_tool_responses)
                        pending_tool_responses = []
                    
                    #Store this ai message for later (Unitl we collect all tool responses)
                    assistant_with_tool_calls = {
                        "role": "assistant",
                        "content": msg['content'],
                        "tool_calls": metadata.get('tool_calls', {})
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
            ai_response = self.model.bind_tools(available_tools).invoke(messages)
        except Exception as e:
            error_msg = f"Error calling Qwen Model: {str(e)}"
            logger.error(error_msg)
            await self.add_to_history("assistant", error_msg, {"error": True})
            return error_msg
        
        #Process response and handle tool calls from ai messages
        tool_results = []
        final_text = []

        ai_message = ai_response.content
        initial_response = ai_message or ""

        #Add initial ai response to history with metadata about tool calls
        tool_calls_metadata = {}
        if ai_response.tool_calls:
            tool_calls_metadata = {
                "has_tool_calls": True,
                "tool_calls": ai_response.tool_calls
            }

        await self.add_to_history("assistant", initial_response, tool_calls_metadata) #first ai message added to the history
        final_text.append(initial_response)

        #Check if tool calls are present and parse the tools
        if ai_response.tool_calls:
            if self.debug:
                logger.info(f"Tool calls requested: {len(ai_response.tool_calls)} tool calls from the llm")

        #Add the assistant's message to the conversation
            messages.append(
                {
                    "role": "assistant",
                    "content": ai_message,
                    "tool_calls": ai_response.tool_calls
                }
            )

            #Process each tool call
            for tool_call in ai_response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['arguments']

                #convert json string to dict if needed
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except Exception as e:
                        logger.warning(f"Failed to parse tool argument as JSON: {tool_args}")
                        tool_args = {}

                if self.debug:
                    logger.info(f"Executing tool: {tool_name} with Arguments {tool_args}")
        
                #Execute tool call on the server
                try:
                    result = await self.session.call_tool(tool_name, tool_args)
                    if hasattr(result, "content") and result.content:
                        tool_content = "\n".join(
                            [part.text if hasattr(part, "text") else str(part) for part in result.content]
                            )
                    else:
                        tool_content = str(result)
                    tool_results.append({"call": tool_name, "result": tool_content})
                    final_text.append(f"\n[Calling tool {tool_name} with args {tool_args}]")

                    if self.debug:
                        result_preview = tool_content[0].text[:100] + "..." if len(tool_content[0].text) > 100 else tool_content[0].text
                        logger.info(f"Tool result preview: {result_preview}")
                    
                    #Add the tool result to the conversation
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_content
                        }
                    )
                    await self.add_to_history("tool", tool_content,
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
                second_response = self.model.bind_tools(available_tools).invoke(messages)

                response_content = second_response.content or ""
                await self.add_to_history("assistant", response_content)
                final_text.append("\n" + response_content)
            except Exception as e:
                error_msg = f"Error getting final response from Qwen3: {str(e)}"
                logger.error(error_msg)
                await self.add_to_history("assistant", error_msg, {"error": True})
                final_text.append(f"\n[Error: {error_msg}]")

        return "\n".join(final_text)
    
    # ===================================================
    # Main Chat Loop
    # ===================================================

    async def chat_loop(self):
        """Welcome to the RAG-AI-MCP Client!"""
        print(f"\n{'='*50}")
        print(f"RAG-AI-MCP Client Connected to: {self.server_name}")
        print(f"{'='*50}")
        print("Type your queries or use these commands:")
        print(" /debug - Toggle debug mode")
        print(" /refresh - Refresh server capabilities")
        print(" /resources - List available resources")
        print(" /resource <uri> - Read a specific resource")
        print(" /prompts - List available prompts")
        print(" /prompt <name> <argument> - Use a specific prompt with string as the argument")
        print(" /tools - List available tools")
        print(" /quit - Exit the Client")

        while True:
            try:
                #Get user query
                query = input("\nQuery: ").strip()

                #Handle commands
                if query.lower() == '/quit':
                    break

                #Toggle debug mode
                elif query.lower() == '/debug':
                    self.debug = not self.debug
                    print(f"\nDebug mode {'enabled' if self.debug else 'disabled'}")
                    continue

                #Refresh server capabilities
                elif query.lower() == '/refresh':
                    await self.refresh_capabilities()
                    print('\n Server capabilities refreshed')
                    continue

                #List Available resources
                elif query.lower() == '/resources':
                    resources = await self.list_resources()
                    print("\nAvailable Resources:")
                    for res in resources:
                        print(f" - {res.uri}")
                        if res.description:
                            print(f"   {res.description}")
                    continue

                #Read content from a resource
                elif query.lower().startswith('/resource '):
                    uri = query[10:].strip()
                    print(f"\nFetching resource: {uri}")
                    content = await self.read_resource(uri)
                    print(f"\nResource Content ({uri}):")
                    print("------------------------------------------------")
                    if len(content) > 500:
                        print(content[:500] + "...")
                        print("(Resource content truncated for display purpose but full content is included in message history)")
                    else:
                        print(content)
                    continue

                #List available Prompts
                elif query.lower() == '/prompts':
                    prompts = await self.list_prompts()
                    print("\nAvailable Prompts:")
                    for prompt in prompts:
                        print(f" - {prompt.name}")
                        if prompt.description:
                            print(f"    {prompt.description}")
                        if prompt.arguments:
                            print(f"    Argument: {', '.join(arg.name for arg in prompt.arguments)}")
                    continue
                
                #Run a specific prompt with arguments
                elif query.lower().startswith("/prompt "):
                    #parse: /prompt name sentence of args
                    parts = query[8:].strip().split(maxsplit=1)
                    if not parts:
                        print("Error: Prompt name required")
                        continue

                    name = parts[0]
                    arguments = {}

                    if len(parts) > 1:
                        arg_text = parts[1]

                        #Get the prompt to check its expected arguments
                        prompt_info = None
                        for prompt in self.available_prompts:
                            if prompt.name == name:
                                prompt_info = prompt
                                break

                        if prompt_info and prompt_info.arguments and len(prompt_info.arguments) > 0:
                            #use the first argument name as the key for the entire sentence
                            arguments[prompt_info.arguments[0].name] = arg_text
                        else:
                            arguments['text'] = arg_text
                    print(f"\nGetting prompt template: {name}")
                    prompt_result = await self.get_prompt(name, arguments)

                    #Process the prompt with the LLM and add to conversation
                    if not self.model:
                        print("Error: LLM model is not initialized. Cannot process prompts.")
                        continue

                    messages = prompt_result.messages

                    #Convert message to OpenAI format and include relevant history
                    llm_messages = []
                    recent_messages = []

                    for msg in reversed(self.message_history[-10:]):
                        if msg['role'] in ['user', 'assistant'] and len(recent_messages) < 5:
                            recent_messages.append({
                                "role": msg.role,
                                "content": msg['content']
                            })

                    #Add recent messages in correct order(Older first)
                    llm_messages.extend(reversed(recent_messages))

                    #Then add the prompt messages
                    for msg in messages:
                        content = msg.content.text if hasattr(msg.content, "text") else str(msg.content)
                        llm_messages.append({
                            "role": msg.role,
                            "content": content
                        })
                        await self.add_to_history(msg.role, content, {"is_prompt_message": True, "prompt_name": name})
                    print("Processing prompt...")

                    try:
                        response = self.model.bind_tools(self.available_tools).invoke(llm_messages)

                        response_content = response.content or ""

                        #Add the prompt and response to the conversation history
                        await self.add_to_history("assistant", response_content)
                        print("\n" + response_content)
                    except Exception as e:
                        error_msg = f"\nError processing prompt with Qwen3: {str(e)}"
                        print(error_msg)
                    continue

                #List Available tools
                elif query.lower() == '/tools':
                    print(f"\nAvailable Tools:")
                    for tool in self.available_tools:
                        print(f"   - {tool.name}")
                        if tool.description:
                            print(f"    {tool.description}")
                    continue

                #process regular queries
                print("\nProcessing query...")
                response = await self.process_query(query)
                print("\n" + response)
            
            except Exception as e:
                print(f"\n Error: {str(e)}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

    # ===========================================
    # Resource CleanUp
    # ===========================================
    async def cleanup(self):
        """Clean up resources"""
        if self.debug:
            logger.info("Cleaning up client resources")
        await self.exit_stack.aclose()

# ===========================================
# Main Function
# ===========================================
async def main():
    """Run the MCP Client"""
    
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
    
    #initialize client
    server_script  = sys.argv[1]
    client = MCPClient()

    #connect to server
    try:
        connected = await client.connect_to_server(server_script)
        if not connected:
            print(f"Failed to connect to server at {server_script}")
            sys.exit(1)
        
        await client.chat_loop()

        #Handle other exceptions
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

    #CleanUp resources
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())


                    





