# Corrections to the first draft

The first draft mixed useful examples with unsupported hardware claims.
These corrections apply to the earlier discussion and materials, not findings
from the current examples.

| Earlier claim | Correction |
| --- | --- |
| GLM-4.7-Flash is an 8B model | It is a 30B-A3B MoE; total and active parameters serve different purposes |
| Weight size below VRAM guarantees a fit | Runtime buffers, cache, other state, and other applications also need memory |
| Cached attention is entirely linear | K/V projection work becomes linear over the sequence; full-attention decode comparisons remain quadratic |
| Q8 cache is always free; never use Q4 | Quality and speed depend on the model, workload, and implementation |
| Q8 cache uses exactly one byte per value | Q8_0 also stores a two-byte scale for each 32-value block |
| Random head correlations demonstrate GQA quality loss | Shared K/V does not force identical attention; quality needs trained-model evaluation |
| FlashAttention makes total memory constant | It avoids the full score matrix; persistent state and other memory remain |
| llama.cpp always silently ignores unsupported cache flags | Current code can auto-enable required FlashAttention or report an error |
| Original GLM-4-9B-Chat has four KV groups | Its configuration specifies two |

The old fixed throughput ranges, RAM-upgrade predictions, and thermal-throttling
claims were not established by measurements. The recorded baseline is retained
with its workload and timing fields rather than presented as a hardware rating.

The NumPy low-bit example is now labeled as a toy representation, not a faithful
implementation of llama.cpp Q4_0. The cached-attention and online-softmax examples
describe their scope explicitly.
