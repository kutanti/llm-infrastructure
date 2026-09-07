# Use an existing Ollama installation

The offline lessons need Python and NumPy only. For the HTTP exercises, you
also need a local Ollama server with `qwen3.5:4b` available.

## Find the server first

Ollama normally uses port 11434. The laptop deployment that produced this
repository's baseline uses 11435.

Try the endpoint for your installation:

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/version
Invoke-RestMethod http://127.0.0.1:11434/api/tags | ConvertTo-Json -Depth 6
```

For the original laptop deployment, substitute `11435` in both commands.
The first shows the server version; the second lists downloaded models.
`/api/ps` lists only models currently loaded in memory.

If the server responds and lists `qwen3.5:4b`, skip to **Run the client**.

## Start Ollama if it is not running

Locate the existing executable:

```powershell
Get-Command ollama -ErrorAction SilentlyContinue
Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
```

If you installed Ollama somewhere else, use that executable's path. On the
original laptop, the portable copy is
`C:\Projects\Qwen-Local\runtime\ollama.exe`; its existing launcher is
`C:\Projects\Qwen-Local\Start-Server.ps1`.

For a standard installation, open a dedicated PowerShell terminal:

```powershell
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_CONTEXT_LENGTH = '4096'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_KV_CACHE_TYPE = 'f16'
$env:OLLAMA_KEEP_ALIVE = '10m'
ollama serve
```

If the executable is not on PATH, replace `ollama serve` with:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve
```

Leave that terminal open. If Ollama reports the address is already in use,
query the existing endpoint instead of starting another server.
These commands retain the installation's configured/default model directory.

On a machine with no Ollama installation, install it from
[ollama.com/download](https://ollama.com/download), then return here.
This repository does not contain or install a runtime.

## Download the model only if missing

In a second terminal, target the same server:

```powershell
$env:OLLAMA_HOST = '127.0.0.1:11434'
ollama pull qwen3.5:4b
```

Use the executable's full path if necessary. The original Q4_K_M package was
about 3.39 GB; the tag can change. The server stores the download for reuse.
If an older runtime does not support the architecture, update that installation.

## Run the client

From the cloned learning repository:

```powershell
python 08_local_client.py --url http://127.0.0.1:11434 --prompt "What is 17 * 23? Reply with only the integer." --save results\first.json
```

For the original deployment on port 11435, omit `--url`.
The examples in `LABS.md` use that default; add the 11434 argument when needed.

The client does not download models, restart servers, or change global settings.
It supplies per-request context, output-token limit, temperature, and thinking
mode. New result files are ignored by Git and existing ones are not overwritten.

Exit an interactive `ollama run` chat with `/bye`. Stop a manually started server
with Ctrl+C in its server terminal. Neither action deletes the model download.
