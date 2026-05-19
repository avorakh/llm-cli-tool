import click
import time
import json
from dotenv import load_dotenv
from llm_client import LLMClient
from config import MODELS, DEFAULT_MODEL
from utils import count_tokens, calculate_cost, format_cost

load_dotenv()

SEPARATOR = "─" * 50


@click.command()
@click.argument("prompt")
@click.option("--model", default=DEFAULT_MODEL, type=click.Choice(list(MODELS.keys())), help="Model to use")
@click.option("--temperature", default=0.7, type=float, help="Sampling temperature (0.0–2.0)")
@click.option("--max-tokens", default=2000, type=int, help="Maximum output tokens")
@click.option("--no-stream", is_flag=True, help="Disable streaming")
@click.option("--json-output", is_flag=True, help="Return metrics as JSON")
def main(prompt, model, temperature, max_tokens, no_stream, json_output):
    """Call an LLM and stream the response with cost tracking."""
    config = MODELS[model]
    client = LLMClient(model)

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
        in_tokens = input_tokens   # estimate; streaming doesn't return usage
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
            "model":          model,
            "input_tokens":   in_tokens,
            "output_tokens":  out_tokens,
            "latency_sec":    round(latency, 3),
            "cost_usd":       round(cost, 6),
            "response":       output_text,
        }
        click.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
