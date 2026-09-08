# Before running a model

Start with a simpler question than GPU performance:

> Given "the cat sat on the", how does a collection of numbers decide what
> comes next, and how did those numbers get there?

The chapters below follow that question from text to prediction to learning.
You need basic Python, not prior machine learning. The equations are accompanied
by small examples; the scripts run on a CPU with NumPy.

## Reading order

| Chapter | Question | Runnable example |
| --- | --- | --- |
| [1. Text and representations](01-text-and-representations.md) | What does a model receive instead of words? | `01_tokens_and_probabilities.py` |
| [2. Learning from mistakes](02-learning-from-mistakes.md) | How does prediction error change a weight? | `02_train_a_small_model.py` |
| [3. Inside a transformer](03-inside-a-transformer.md) | How does a token use the rest of the sentence? | `03_transformer_block.py` |
| [4. From training to answers](04-from-training-to-answers.md) | How does a pretrained model become an assistant? | Experiments in the chapter |

Run these from the repository root:

```powershell
python foundations\01_tokens_and_probabilities.py
python foundations\02_train_a_small_model.py
python foundations\03_transformer_block.py
```

The second script **actually learns weights** from a tiny synthetic corpus.
The third exposes a transformer's forward computation using random weights;
it does not generate meaningful language. Keeping those examples separate
makes it easier to see which part of the problem each solves.

The [exercises](EXERCISES.md) ask you to predict an output or change an input
before looking at the answer. Each chapter also contains a worked calculation.

## What you should be able to explain afterward

Take the prompt `"red blue"` through a model. Identify token IDs, embedding
rows, hidden states, logits, probabilities, and the chosen next token. Explain
which values are weights, which are temporary state, and which get updated
during training.

Then explain why generating a longer answer costs more time, why a longer
conversation needs more state, and why making weights smaller does not
necessarily make the model more accurate.

That leads into [the laptop inference investigation](../GUIDE.md). Its
102-second first response is a systems problem built on these computations.
