import json
import time
import uuid
from datetime import datetime, timezone

import click
from dotenv import load_dotenv

from llm_cli_tool.config import DEFAULT_MODEL, MODELS
from llm_cli_tool.llm_client import LLMClient
from llm_cli_tool.schemas import (
    JSON_SCHEMAS,
    JsonResponseValidationError,
    get_json_schema_instruction,
    validate_json_response,
)
from llm_cli_tool.usage_store import UsageEvent, UsageStore
from llm_cli_tool.utils import calculate_cost, count_tokens, format_cost

load_dotenv()

SEPARATOR = "─" * 50
EXIT_COMMANDS = {"exit", "quit", "q"}


@click.command()
@click.argument("prompt", required=False)
@click.option("--model", default=DEFAULT_MODEL, type=click.Choice(list(MODELS.keys())), help="Model to use")
@click.option("--temperature", default=0.7, type=float, help="Sampling temperature (0.0-2.0)")
@click.option("--max-tokens", default=2000, type=int, help="Maximum output tokens")
@click.option("--no-stream", is_flag=True, help="Disable streaming")
@click.option("--json-response", is_flag=True, help="Request a JSON object response from the model")
@click.option(
    "--json-schema",
    type=click.Choice(JSON_SCHEMAS),
    help="Validate a JSON response against a built-in schema",
)
@click.option("--cost-report", is_flag=True, help="Show cost usage for the current week")
@click.option("--json-output", is_flag=True, help="Return metrics as JSON")
@click.option("--chat", is_flag=True, help="Keep a conversational history across turns")
def main(prompt, model, temperature, max_tokens, no_stream, json_response, json_schema, cost_report, json_output, chat):
    """Call an LLM and stream the response with cost tracking."""
    config = MODELS[model]
    client = LLMClient(model)
    usage_store = UsageStore()
    session_id = str(uuid.uuid4())

    _validate_json_options(model, json_response, json_schema, cost_report, chat, prompt)

    if cost_report:
        summary = usage_store.get_weekly_summary()
        _echo_cost_report(summary, json_output)
        return

    if chat:
        run_chat_session(
            client,
            usage_store,
            session_id,
            model,
            temperature,
            max_tokens,
            no_stream,
            json_response,
            json_schema,
            json_output,
            prompt,
        )
        return

    if prompt is None:
        raise click.UsageError("Prompt is required unless --chat or --cost-report is enabled")

    messages = _build_messages(prompt, json_schema)
    input_tokens = count_tokens("\n".join(message["content"] for message in messages), model)
    click.echo(f"Using model: {model}")
    click.echo(f"Input tokens: {input_tokens}\n")

    start = time.perf_counter()
    response_format = _build_response_format(json_response)

    if no_stream or json_response:
        output_text, in_tokens, out_tokens = client.complete_messages(
            messages,
            temperature,
            max_tokens,
            response_format=response_format,
        )
        response_payload = _parse_response_payload(output_text, json_response, json_schema)
        _echo_response(response_payload)
    else:
        output_text = ""
        for token in client.stream(prompt, temperature, max_tokens):
            output_text += token
            click.echo(token, nl=False)
        click.echo()
        in_tokens = input_tokens  # estimate; streaming doesn't return usage
        out_tokens = count_tokens(output_text, model)
        response_payload = output_text

    latency = time.perf_counter() - start
    cost = calculate_cost(in_tokens, out_tokens, config["cost"]["input"], config["cost"]["output"])
    _log_usage(
        usage_store=usage_store,
        session_id=session_id,
        model=model,
        provider=config["provider"],
        in_tokens=in_tokens,
        out_tokens=out_tokens,
        cost=cost,
        latency=latency,
        streamed=not (no_stream or json_response),
        chat_turn=False,
        json_response=json_response,
    )

    click.echo(f"\n{SEPARATOR}")
    click.echo(f"Latency:       {latency:.2f}s")
    click.echo(f"Output tokens: {out_tokens}")
    click.echo(f"Cost:          {format_cost(cost)}")
    click.echo(f"Total tokens:  {in_tokens + out_tokens}")

    if json_output:
        result = {
            "model": model,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "latency_sec": round(latency, 3),
            "cost_usd": round(cost, 6),
            "response": response_payload,
        }
        click.echo(json.dumps(result, indent=2))


def run_chat_session(
    client,
    usage_store,
    session_id,
    model,
    temperature,
    max_tokens,
    no_stream,
    json_response,
    json_schema,
    json_output,
    prompt,
):
    history = _initialize_history(json_schema)
    click.echo(f"Using model: {model}")
    click.echo("Chat mode. Type /exit, /quit, or /q to stop.\n")

    if prompt:
        history.append({"role": "user", "content": prompt})
        _complete_chat_turn(
            client,
            usage_store,
            session_id,
            model,
            temperature,
            max_tokens,
            no_stream,
            json_response,
            json_schema,
            json_output,
            history,
        )

    while True:
        user_input = click.prompt("> ", prompt_suffix="", default="", show_default=False).strip()
        if not user_input:
            continue
        if user_input.lstrip("/").lower() in EXIT_COMMANDS:
            break
        history.append({"role": "user", "content": user_input})
        _complete_chat_turn(
            client,
            usage_store,
            session_id,
            model,
            temperature,
            max_tokens,
            no_stream,
            json_response,
            json_schema,
            json_output,
            history,
        )


def _complete_chat_turn(
    client,
    usage_store,
    session_id,
    model,
    temperature,
    max_tokens,
    no_stream,
    json_response,
    json_schema,
    json_output,
    history,
):
    config = MODELS[model]
    input_tokens = count_tokens("\n".join(message["content"] for message in history), model)
    click.echo(f"Input tokens: {input_tokens}")

    start = time.perf_counter()
    response_format = _build_response_format(json_response)

    if no_stream or json_response:
        output_text, in_tokens, out_tokens = client.complete_messages(
            history,
            temperature,
            max_tokens,
            response_format=response_format,
        )
        response_payload = _parse_response_payload(output_text, json_response, json_schema)
        _echo_response(response_payload)
    else:
        output_text = ""
        for token in client.stream_messages(history, temperature, max_tokens):
            output_text += token
            click.echo(token, nl=False)
        click.echo()
        in_tokens = input_tokens
        out_tokens = count_tokens(output_text, model)
        response_payload = output_text

    history.append({"role": "assistant", "content": _serialize_response_payload(response_payload)})

    latency = time.perf_counter() - start
    cost = calculate_cost(in_tokens, out_tokens, config["cost"]["input"], config["cost"]["output"])
    _log_usage(
        usage_store=usage_store,
        session_id=session_id,
        model=model,
        provider=config["provider"],
        in_tokens=in_tokens,
        out_tokens=out_tokens,
        cost=cost,
        latency=latency,
        streamed=not (no_stream or json_response),
        chat_turn=True,
        json_response=json_response,
    )

    click.echo(f"{SEPARATOR}")
    click.echo(f"Latency:       {latency:.2f}s")
    click.echo(f"Output tokens: {out_tokens}")
    click.echo(f"Cost:          {format_cost(cost)}")
    click.echo(f"Total tokens:  {in_tokens + out_tokens}\n")

    if json_output:
        result = {
            "model": model,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "latency_sec": round(latency, 3),
            "cost_usd": round(cost, 6),
            "response": response_payload,
        }
        click.echo(json.dumps(result, indent=2))


def _validate_json_options(model, json_response, json_schema, cost_report, chat, prompt):
    if json_schema and not json_response:
        raise click.UsageError("--json-schema requires --json-response")

    if json_response and MODELS[model]["provider"] != "openai":
        raise click.UsageError("--json-response is currently supported only for OpenAI models")

    if cost_report and chat:
        raise click.UsageError("--cost-report cannot be used with --chat")

    if cost_report and prompt is not None:
        raise click.UsageError("--cost-report does not accept a prompt")

    if cost_report and (json_response or json_schema):
        raise click.UsageError("--cost-report cannot be used with JSON response options")


def _build_response_format(json_response):
    if json_response:
        return {"type": "json_object"}
    return None


def _build_messages(prompt, json_schema):
    messages = [{"role": "user", "content": prompt}]
    if json_schema:
        messages.insert(0, {"role": "system", "content": get_json_schema_instruction(json_schema)})
    return messages


def _initialize_history(json_schema):
    if not json_schema:
        return []
    return [{"role": "system", "content": get_json_schema_instruction(json_schema)}]


def _parse_response_payload(output_text, json_response, json_schema):
    if not json_response:
        return output_text

    try:
        return validate_json_response(output_text, json_schema)
    except JsonResponseValidationError as exc:
        raise click.ClickException(str(exc)) from exc


def _serialize_response_payload(payload):
    if isinstance(payload, dict):
        return json.dumps(payload)
    return payload


def _echo_response(payload):
    if isinstance(payload, dict):
        click.echo(json.dumps(payload, indent=2))
        return
    click.echo(payload)


def _log_usage(usage_store, session_id, model, provider, in_tokens, out_tokens, cost, latency, streamed, chat_turn, json_response):
    usage_store.log_usage(
        UsageEvent(
            session_id=session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            model=model,
            provider=provider,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=cost,
            latency_sec=latency,
            streamed=streamed,
            chat_turn=chat_turn,
            request_mode="json_response" if json_response else "text",
        )
    )


def _echo_cost_report(summary, json_output):
    if json_output:
        payload = {
            "total_cost_usd": round(summary.total_cost_usd, 6),
            "total_calls": summary.total_calls,
            "input_tokens": summary.total_input_tokens,
            "output_tokens": summary.total_output_tokens,
            "top_model": summary.top_model,
            "top_model_share_pct": round(summary.top_model_share, 2),
        }
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo(f"Total this week: {format_cost(summary.total_cost_usd)}")
    click.echo(f"Total calls:     {summary.total_calls}")
    click.echo(f"Input tokens:    {summary.total_input_tokens}")
    click.echo(f"Output tokens:   {summary.total_output_tokens}")

    if summary.top_model is None:
        click.echo("Top model:       n/a")
        return

    click.echo(f"Top model:       {summary.top_model} ({summary.top_model_share:.0f}%)")


if __name__ == "__main__":
    main()
