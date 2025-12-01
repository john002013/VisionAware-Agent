from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import os
from langchain.agents import create_tool_calling_agent, AgentExecutor
from visual_tool import visual_tool

llm = ChatOpenAI(
    openai_api_key="Your API Key here (OpenAI, OpenRouter, Claude, Gemini)",
                             base_url ="https://models.github.ai/inference",
                             model_name="openai/gpt-4o",
                             temperature=0.1
)
class visualAgentResponse(BaseModel):
    detected_object: str
    tools_used: list[str]

parser = PydanticOutputParser(pydantic_object = visualAgentResponse)

System_prompt = ChatPromptTemplate.from_messages(
    [
        (

    "system",
    """
You are a Visual Agent.

Your task is to analyze what the webcam sees using object detection and return a structured summary using the appropriate tool.

You have access to the following tool:
- `visual_tool` → uses a camera and object detection model to identify visible objects in real time and return a summary of detected items and their counts (e.g., "3 persons, 1 bottle").

Instructions:

- ALWAYS use the `visual_tool` when asked to identify what is in the camera view or what is visible.
- Do NOT generate your own object list manually. Use the tool to detect them.
- The tool returns a structured response with two fields:
  - `detected_object`: a string like "3 persons, 2 bottles"
  - `tools_used`: a list, which must always include `"visual_tool"`
- DO NOT respond directly without using the tool.
- After using the tool, give a friendly and **brief** reply, like:
  - "I see 3 persons and 1 laptop."
  - "Looks like there's a chair and a bottle in view."
- Only describe what the tool detected. Don’t add anything extra.
- If nothing is detected, say something like:
  - "I couldn’t spot anything just now."


Examples:

User: "What do you see?"
→ use: `visual_tool()`
→ reply: "I see 2 persons and a chair."

User: "Can you check what's around?"
→ use: `visual_tool()`
→ reply: "Looks like there’s a bottle and a book nearby."

Now wait for user input and always use the `visual_tool` before responding.
""",
),

("placeholder", "{chat_history}"),
("human","{query}"),
("placeholder", "{agent_scratchpad}"),

]
).partial(format_instructions=parser.get_format_instructions())

tools = [visual_tool]
agent = create_tool_calling_agent(
    llm = llm,
    prompt=System_prompt,
    tools=tools
)

visual_agent = AgentExecutor(agent = agent, tools = tools, verbose=True)

query = input("Want to get deep insight about your environment?")
raw_response= visual_agent.invoke({"query": query})
response = raw_response.get("output")
print(response)
