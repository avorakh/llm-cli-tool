# LLM CLI Tool

The tool accepts a prompt as a CLI argument and streams the response back to your terminal in real-time, displaying token counts and cost at the end of each call.

## Setup

Create a virtual environment:

```sh
python3 -m venv venv
```

Activate it:

```sh
source venv/bin/activate
```

Install the project in editable mode:

```sh
pip install -e .
```

## Run And Test

Basic call with streaming:

```sh
llm-cli "What is tokenization?"
```

Try a different model:

```sh
llm-cli --model gpt-4o "Explain attention mechanisms"
```

Disable streaming and get the full response at once:

```sh
llm-cli --no-stream "List 3 LLM concepts"
```

Get metrics as JSON:

```sh
llm-cli --json-output "Explain RAG in one sentence"
```

Start an interactive chat session and keep previous turns in context:

```sh
llm-cli --chat
> explain tokenization
...
> give me a Python example
...
> /exit
```

You can also run the package directly with `python -m llm_cli_tool`.

Deactivate the virtual environment when you are done:

```sh
deactivate
```
