import json
import time

import click
from dotenv import load_dotenv

from llm_cli_tool.config import DEFAULT_MODEL, MODELS
from llm_cli_tool.llm_client import LLMClient
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
@click.option("--json-output", is_flag=True, help="Return metrics as JSON")
@click.option("--chat", is_flag=True, help="Keep a conversational history across turns")
def main(prompt, model, temperature, max_tokens, no_stream, json_output, chat):
    """Call an LLM and stream the response with cost tracking."""
    config = MODELS[model]
    client = LLMClient(model)

    if chat:
        run_chat_session(client, model, temperature, max_tokens, no_stream, json_output, prompt)
        return

    if prompt is None:
        raise click.UsageError("Prompt is required unless --chat is enabled")

    input_tokens = count_tokens(prompt, model)
    click.echo(f"Using model: {model}")
    click.echo(f"Input tokens: {input_tokens}\n")

    start = time.perf_counter()

    if no_stream:
        output_text, in_tokens, out_tokens = client.complete(prompt, temperature, max_tokens)
        click.echo(output_text)
    else:
        output_text = ""
        for token in client.stream(prompt, temperature, max_tokens):
            output_text += token
            click.echo(token, nl=False)
        click.echo()
        in_tokens = input_tokens  # estimate; streaming doesn't return usage
        out_tokens = count_tokens(output_text, model)

    latency = time.perf_counter() - start
    cost = calculate_cost(in_tokens, out_tokens, config["cost"]["input"], config["cost"]["output"])

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
            "response": output_text,
        }
        click.echo(json.dumps(result, indent=2))


def run_chat_session(client, model, temperature, max_tokens, no_stream, json_output, prompt):
    history = []
    click.echo(f"Using model: {model}")
    click.echo("Chat mode. Type /exit, /quit, or /q to stop.\n")

    if prompt:
        history.append({"role": "user", "content": prompt})
        _complete_chat_turn(client, model, temperature, max_tokens, no_stream, json_output, history)

    while True:
        user_input = click.prompt("> ", prompt_suffix="", default="", show_default=False).strip()
        if not user_input:
            continue
        if user_input.lstrip("/").lower() in EXIT_COMMANDS:
            break
        history.append({"role": "user", "content": user_input})
        _complete_chat_turn(client, model, temperature, max_tokens, no_stream, json_output, history)


def _complete_chat_turn(client, model, temperature, max_tokens, no_stream, json_output, history):
    config = MODELS[model]
    input_tokens = count_tokens("\n".join(message["content"] for message in history), model)
    click.echo(f"Input tokens: {input_tokens}")

    start = time.perf_counter()

    if no_stream:
        output_text, in_tokens, out_tokens = client.complete_messages(history, temperature, max_tokens)
        click.echo(output_text)
    else:
        output_text = ""
        for token in client.stream_messages(history, temperature, max_tokens):
            output_text += token
            click.echo(token, nl=False)
        click.echo()
        in_tokens = input_tokens
        out_tokens = count_tokens(output_text, model)

    history.append({"role": "assistant", "content": output_text})

    latency = time.perf_counter() - start
    cost = calculate_cost(in_tokens, out_tokens, config["cost"]["input"], config["cost"]["output"])

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
            "response": output_text,
        }
        click.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
