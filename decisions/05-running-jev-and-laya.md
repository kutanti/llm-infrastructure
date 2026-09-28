# 5. Running Jev and Laya

The request from lesson 2 works with three backends: TypeSafe's hosted API, a
`laya-serve` process on your machine, and Laya loaded into Python.
[`ask.py`](ask.py) sends the course questions to any of them. It prints the request and
stops unless you add `--live`.

This course shows no real output from either model. Run these commands to see real
answers, and record the model version beside them.

## Look before sending

```powershell
python -m decisions.ask --jev
```

The dry run prints the target URL, the headers with the key shown only as the name of
its environment variable, and the JSON body. Nothing is sent. Add `--examples` to see
the first of the 15 labeled tickets instead of the course ticket.

## Jev: the hosted API

You need a TypeSafe API key. Put it in the environment of the current shell. The script
never accepts a key as an argument, because arguments end up in shell history and
process listings:

```powershell
$env:TYPESAFE_API_KEY = Read-Host -MaskInput "TypeSafe API key"
python -m decisions.ask --jev --live
```

`-MaskInput` needs PowerShell 7.1 or later. In a POSIX shell, use
`read -rs TYPESAFE_API_KEY; export TYPESAFE_API_KEY`.

The script prints one row per question:

```text
question          type          answer   top p  band     backend confidence
```

`answer` is the choice, `P(yes)` for a noul, or the expected level for a score. `top p`
is the statistic from lesson 3, and `band` applies the lesson 4 thresholds (`--act-at`,
`--confirm-at`). `backend confidence` is Jev's own `confidence` field. Noul answers show
`-` there because Jev returns none for them.

The line `Answered by:` repeats the response's `model` field. Jev's schema says it "may
differ from the alias supplied in the request". Thresholds belong to one model, so pin a
model name once you have tuned them. The official SDK lists the available names:

```powershell
python -m pip install typesafe-sdk==0.7.2
```

```python
from typesafe_sdk import TypeSafeClient

with TypeSafeClient() as client:  # reads TYPESAFE_API_KEY
    for model in client.models.list().models:
        print(model.name, model.release_date, model.description)
```

Then pass the name with `--model`. The SDK also sends questions directly. It accepts the
course's plain dictionaries:

```python
from typesafe_sdk import TypeSafeClient
from decisions.questions import TICKET, TRIAGE_QUESTIONS

with TypeSafeClient() as client:
    response = client.system_one(state=TICKET, questions=TRIAGE_QUESTIONS, model="jev-latest")

print(response.model, response.choices["department"].probabilities)
print(response.nouls["refund_requested"].noul, response.usage.input_tokens)
```

`usage.input_tokens` is what Jev bills. Keep the `x-typesafe-request-id` response header,
which `ask.py` saves as `request_id`, if you need to report a problem to TypeSafe.

## Laya in the same process

Laya needs PyTorch and `transformers`. Install it in a separate virtual environment so
this repository's NumPy-only environment stays small. On Windows:

```powershell
python -m venv decisions\.venv
.\decisions\.venv\Scripts\python.exe -m pip install laya==0.3.21
.\decisions\.venv\Scripts\python.exe -m decisions.ask --laya --live --allow-download
```

Laya's README describes CPU-only and GPU-specific PyTorch builds for each platform. On
Linux or macOS, the environment's Python is `decisions/.venv/bin/python`.

Without `--allow-download`, the script sets `HF_HUB_OFFLINE=1`, so only weights already
in the Hugging Face cache can load. The first run needs the flag to download the
checkpoint the router chooses. With `--examples`, the Spanish ticket may route to
`laya-multilingual`, which is a second download.

The response's `model` is always `laya-rl-agent`. The checkpoint that answered is in
`routing`, which the script prints as `routed to english (...)` together with the reason.
Pass `--model english`, `multilingual`, or `typed-decisions` to skip the router.

Without `--revision`, the current weights on the Hub load, and they can change. Pin a
Hub commit with `--revision <commit SHA>`. Alternatively, `--revision reviewed` uses the
commits pinned inside the `laya` package. The first in-process call also includes
building the model, so lesson 6 reports it apart from the later calls.

## Laya over HTTP

`laya-serve` answers `POST /v1/systemone` like Jev does:

```powershell
.\decisions\.venv\Scripts\python.exe -m pip install "laya[serve]==0.3.21"
$env:LAYA_HOST = '127.0.0.1'
$env:LAYA_MODELS = 'english'
.\decisions\.venv\Scripts\laya-serve.exe
```

- **`LAYA_HOST` defaults to `0.0.0.0`**, which accepts connections from other machines.
  Set `127.0.0.1` unless others need access. If they do, set `LAYA_API_KEY` and put TLS
  in front of the server.
- **`LAYA_MODELS` limits what loads at startup.** By default the server builds every
  checkpoint when it starts, downloading any that are not cached.
- `GET http://127.0.0.1:8000/health` lists the loaded checkpoints and the revision each
  was loaded from.

In a second terminal, in this repository's environment:

```powershell
python -m decisions.ask --url http://127.0.0.1:8000 --live
```

If the server has `LAYA_API_KEY` set, set the same variable in this shell and add
`--api-key-env LAYA_API_KEY`. `ask.py` accepts plain `http` only for `127.0.0.1`, `::1`,
and `localhost`; any other host must use `https`. It does not follow redirects, so a
key cannot be forwarded to a different host.

Laya's HTTP documentation says Jev clients, including `typesafe-sdk`, keep working when
pointed at `laya-serve`; the SDK reads its base URL from `TYPESAFE_BASE_URL`. This course
did not test that path. A client sends its key to whatever base URL it has, so give it
the server's key and never your TypeSafe key. The SDK also drops fields that are not in
Jev's schema, such as `answer_confidence` and `routing`.

## What differs between the backends

| | Jev | Laya (`laya-serve` or in-process) |
| --- | --- | --- |
| `model` in the request | Required; selects the model | A Laya checkpoint name selects it. Anything else, including `jev-latest`, lets the router choose |
| `model` in the response | The model that answered | Always `laya-rl-agent`; read `routing.model` |
| `confidence` | Formula not published | Normalized entropy for choice and score, `max(p)` for noul; `answer_confidence` is `max(p)` |
| Instructions | Optional | Required |
| Timing | `ask.py` measures wall time | The server also sends `X-Inference-Time-Ms`, which `ask.py` saves as `server_inference_ms` |
| Cost | Billed by `usage.input_tokens` | Your hardware; `usage.input_tokens` is still reported |

## Errors

On an HTTP error, `ask.py` prints the status, a hint, and the start of the response body,
which usually names the problem. For example, 401 means the key is missing or wrong, and
422 names the question that failed validation. For 429 and 503, wait and retry.
`laya-serve` returns 503 when `LAYA_MAX_CONCURRENT` requests are already running; it
refuses extra requests rather than queuing them.

## Save what you ran

```powershell
python -m decisions.ask --jev --live --save results\decisions\jev-single.json
```

The report records the target, the requested and answering model, the Laya version and
revision, the thresholds, the questions, every ticket and raw response, and the timings.
It never contains the key. `--save` refuses to replace an existing file, and `results/`
is ignored by Git. The report does contain the ticket text, so keep real customer data
out of it.

References: [TypeSafe Python SDK, v0.7.2](https://github.com/typesafe-ai/typesafe-sdk-python/tree/v0.7.2),
[Laya HTTP API, v0.3.21](https://github.com/NandhaKishorM/laya/blob/v0.3.21/docs/http-api.md),
[Laya README: installation](https://github.com/NandhaKishorM/laya/blob/v0.3.21/README.md#installation-details).
