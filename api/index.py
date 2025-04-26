import os
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.usage import Usage
from .utils.tools import get_current_weather
from .utils.prompt import ClientMessage, convert_to_openai_messages
from typing import List, Dict, Any, cast, AsyncGenerator
import duckdb

import json

load_dotenv(".env.local")

app = FastAPI()

# Create a connection (in-memory by default)
conn = duckdb.connect("test.db")

# Initialize PydanticAI Agent with OpenAI model
# model = GroqModel("meta-llama/llama-4-maverick-17b-128e-instruct")
model = GeminiModel("gemini-2.5-flash-preview-04-17")
agent = Agent(
    model,
    system_prompt="You are a helpful assistant.",
)


# Register the weather tool with the agent
@agent.tool
def get_current_weather_tool(ctx: RunContext[None], latitude: float, longitude: float):
    print("get_current_weather_tool")
    return get_current_weather(latitude, longitude)


def run_duckdb_query(query: str):
    print("run_chdb_query")
    cur = conn.cursor()
    cur.execute(query)
    result = cur.fetchall()
    cur.close()
    return result


@agent.tool
def get_duckdb_schema_tool(ctx: RunContext[None]):
    return get_schema()


def get_schema():
    query = "SELECT database, table, name AS column_name, type AS column_type FROM system.columns WHERE database NOT IN ('SYSTEM', 'INFORMATION_SCHEMA', 'information_schema', 'system') ORDER BY database, table, position;"

    rows = run_duckdb_query(query)
    schema = {}
    for table, name, col_type in rows:
        schema.setdefault(table, []).append(f"{name}:{col_type}")

    # Format the schema information into a compact string
    formatted = "\n".join(
        f"{table}: {', '.join(cols)}" for table, cols in schema.items()
    )
    print(formatted)
    return formatted


@agent.tool
def run_chdb_query_tool(ctx: RunContext[None], query: str):
    print("run_chdb_query_tool")
    try:
        return run_duckdb_query(query)
    except Exception as e:
        return str(e)


# Pydantic model for incoming chat requests
class ChatRequest(BaseModel):
    messages: List[ClientMessage]


async def format_sse_event(data: str) -> str:
    return f"{data}\n"


class QueryRequest(BaseModel):
    query: str


@app.post("/api/query")
async def handle_query_data(request: QueryRequest):
    print("handle_query_data")
    try:
        return run_duckdb_query(request.query)
    except Exception as e:
        stacktrace = traceback.format_exc()
        return {"error": str(e), "stacktrace": stacktrace}


@app.post("/api/chat")
async def handle_chat_data(request: ChatRequest, protocol: str = Query("data")):
    prompt = [r.model_dump_json() for r in request.messages]
    print(prompt[0])

    async def event_stream() -> AsyncGenerator[str, None]:
        print("event stream")
        draft_tool_calls = []
        draft_tool_calls_index = -1

        async with agent.iter(prompt) as run:
            async for node in run:
                if Agent.is_user_prompt_node(node):
                    print(f"=== UserPromptNode: {node.user_prompt} ===")
                elif Agent.is_model_request_node(node):
                    print("=== ModelRequestNode: streaming partial request tokens ===")
                    async with node.stream(run.ctx) as request_stream:
                        async for event in request_stream:
                            if isinstance(event, PartStartEvent):
                                print(
                                    f"[Request] Starting part {event.index}: {event.part!r}"
                                )
                                if (
                                    isinstance(event.part, TextPart)
                                    and event.part.content
                                ):
                                    yield await format_sse_event(
                                        f"0:{json.dumps(event.part.content)}"
                                    )

                            elif isinstance(event, PartDeltaEvent):
                                if isinstance(event.delta, TextPartDelta):
                                    print(
                                        f"[Request] Part {event.index} text delta: {event.delta.content_delta!r}"
                                    )
                                    if event.delta.content_delta:
                                        yield await format_sse_event(
                                            f"0:{json.dumps(event.delta.content_delta)}"
                                        )
                                elif isinstance(event.delta, ToolCallPartDelta):
                                    print(
                                        f"[Request] Part {event.index} args_delta={event.delta.args_delta}"
                                    )
                                    if event.delta.tool_call_id is not None:
                                        draft_tool_calls_index += 1
                                        draft_tool_calls.append(
                                            {
                                                "id": event.delta.tool_call_id,
                                                "name": event.delta.tool_name_delta
                                                or "",
                                                "arguments": "",
                                            }
                                        )
                                        yield await format_sse_event(
                                            f"b:{json.dumps({'toolCallId': event.delta.tool_call_id, 'toolName': event.delta.tool_name_delta})}"
                                        )
                                    elif event.delta.args_delta:
                                        if isinstance(event.delta.args_delta, str):
                                            draft_tool_calls[draft_tool_calls_index][
                                                "arguments"
                                            ] += event.delta.args_delta
                                            yield await format_sse_event(
                                                f"c:{json.dumps({'toolCallId': draft_tool_calls[draft_tool_calls_index]['id'], 'argsTextDelta': event.delta.args_delta})}"
                                            )
                                        elif isinstance(event.delta.args_delta, dict):
                                            draft_tool_calls[draft_tool_calls_index][
                                                "arguments"
                                            ] = json.dumps(event.delta.args_delta)
                                            yield await format_sse_event(
                                                f"c:{json.dumps({'toolCallId': draft_tool_calls[draft_tool_calls_index]['id'], 'argsTextDelta': json.dumps(event.delta.args_delta)})}"
                                            )
                            elif isinstance(event, FinalResultEvent):
                                print(
                                    f"[Result] The model produced a final output (tool_name={event.tool_name})"
                                )
                                if event.tool_name:
                                    print("draft_tool_calls", draft_tool_calls)
                                    pass
                                    # for tool_call in draft_tool_calls:
                                    #     yield await format_sse_event(
                                    #         f"9:{json.dumps({'toolCallId': tool_call['id'], 'toolName': tool_call['name'], 'args': json.loads(tool_call['arguments'])})}"
                                    #     )

                                    #     tool_result = get_current_weather_tool(
                                    #         **json.loads(tool_call["arguments"])
                                    #     )
                                    #     yield await format_sse_event(
                                    #         f"a:{json.dumps({'toolCallId': tool_call['id'], 'result': tool_result})}"
                                    #     )

                                yield await format_sse_event(
                                    f"d:{json.dumps({'finishReason': 'tool-calls' if len(draft_tool_calls) > 0 else 'stop', 'usage': {'promptTokens': 0, 'completionTokens': 0}})}"
                                )

                elif Agent.is_call_tools_node(node):
                    print(
                        "=== CallToolsNode: streaming partial response & tool usage ==="
                    )
                    async with node.stream(run.ctx) as handle_stream:
                        async for event in handle_stream:
                            if isinstance(event, FunctionToolCallEvent):
                                print(
                                    f"[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args}"
                                )
                            elif isinstance(event, FunctionToolResultEvent):
                                print(
                                    f"[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}"
                                )
                elif Agent.is_end_node(node):
                    print(f"=== Final Agent Output: {run.result} ===")

    response = StreamingResponse(event_stream(), media_type="text/event-stream")
    response.headers["x-vercel-ai-data-stream"] = "v1"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    return response
