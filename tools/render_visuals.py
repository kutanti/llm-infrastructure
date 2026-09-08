"""Generate the course's original diagrams/animations; no external service needed."""

from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "animations"
WIDTH, HEIGHT = 1000, 570
BG = "#f8fafc"
INK = "#172033"
MUTED = "#475569"
BLUE = "#2563eb"
TEAL = "#0f766e"
ORANGE = "#c2410c"
LIGHT_BLUE = "#dbeafe"
LIGHT_TEAL = "#ccfbf1"
LIGHT_ORANGE = "#ffedd5"
GREY = "#e2e8f0"


def font(size):
    return ImageFont.load_default(size=size)


def text(draw, xy, value, size=22, fill=INK):
    face = font(size)
    left, top, right, bottom = draw.textbbox(xy, value, font=face)
    if left < 0 or top < 0 or right > WIDTH or bottom > HEIGHT:
        raise ValueError(f"Text extends outside the image: {value!r}")
    draw.text(xy, value, font=face, fill=fill)


def canvas(title, subtitle, step):
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    text(draw, (32, 24), title, 30)
    text(draw, (32, 70), subtitle, 19, MUTED)
    draw.line((32, 108, 968, 108), fill=GREY, width=2)
    text(draw, (32, 540), step, 16, MUTED)
    return image, draw


def box(draw, rect, label, fill=GREY, outline=MUTED, size=20):
    draw.rounded_rectangle(rect, radius=9, fill=fill, outline=outline, width=2)
    left, top, right, bottom = rect
    bounds = draw.textbbox((0, 0), label, font=font(size))
    tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
    text(draw, ((left + right - tw) / 2, (top + bottom - th) / 2 - bounds[1]),
         label, size)


def arrow(draw, start, end, fill=BLUE, width=3):
    draw.line((start, end), fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    points = [end]
    for offset in (-0.5, 0.5):
        points.append((end[0] - 12 * math.cos(angle + offset),
                       end[1] - 12 * math.sin(angle + offset)))
    draw.polygon(points, fill=fill)


def save(name, frames, durations, poster_index):
    if len(frames) != len(durations) or any(d <= 0 for d in durations):
        raise ValueError("Every animation frame must have a positive duration")
    OUT.mkdir(parents=True, exist_ok=True)
    frames[poster_index].save(OUT / f"{name}.png")
    # A shared palette prevents frame-to-frame changes in the background colors.
    palette = frames[poster_index].quantize(colors=128)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    indexed[0].save(
        OUT / f"{name}.gif", save_all=True, append_images=indexed[1:],
        duration=durations, loop=0, disposal=2, optimize=False,
    )
    with Image.open(OUT / f"{name}.gif") as result:
        assert result.n_frames == len(frames)
        assert result.size == (WIDTH, HEIGHT)
    print(f"{name}: {len(frames)} frames; {(OUT / f'{name}.gif').stat().st_size:,} bytes")


def gradient_descent():
    frames = []
    weights = [1.0]
    for _ in range(7):
        weights.append(weights[-1] - 0.1 * (2 * weights[-1] - 6) * 2)
    assert all(abs(w - 3) < abs(weights[i] - 3) for i, w in enumerate(weights[1:]))
    def point(w):
        return (70 + (w / 4) * 520, 460 - (0.5 * (2 * w - 6)**2 / 20) * 290)
    for step, weight in enumerate(weights):
        image, draw = canvas(
            "Gradient descent: change the weight, reduce the loss",
            "One scalar model: prediction = 2w; target = 6; learning rate = 0.1",
            f"Update {step} of 7 | This is a calculated example, not a recorded training run.",
        )
        draw.line((70, 145, 70, 460, 605, 460), fill=MUTED, width=2)
        for tick in range(0, 21, 5):
            y = 460 - tick / 20 * 290
            draw.line((65, y, 70, y), fill=MUTED)
            text(draw, (32, y - 10), str(tick), 17, MUTED)
        points = [point(i * 4 / 160) for i in range(161)]
        draw.line(points, fill=BLUE, width=4)
        for tick in range(5):
            x, _ = point(tick)
            draw.line((x, 460, x, 466), fill=MUTED)
            text(draw, (x - 5, 474), str(tick), 18)
        text(draw, (66, 118), "loss", 18)
        text(draw, (614, 474), "w", 18)
        for previous in weights[:step]:
            x, y = point(previous)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=MUTED)
        x, y = point(weight)
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=ORANGE)
        loss = 0.5 * (2 * weight - 6)**2
        gradient = (2 * weight - 6) * 2
        text(draw, (650, 153), f"w = {weight:.4f}", 26)
        text(draw, (650, 204), f"prediction = {2 * weight:.4f}", 21)
        text(draw, (650, 247), f"loss = {loss:.4f}", 21)
        text(draw, (650, 290), f"gradient = {gradient:.4f}", 21)
        text(draw, (650, 358), "Next update:", 21, MUTED)
        text(draw, (650, 392), "w - 0.1 * gradient", 22, TEAL)
        frames.append(image)
    save("gradient-descent", frames, [1100] * 7 + [2600], 2)


def kv_cache():
    frames = []
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    for count in range(3, 7):
        for phase in ("project", "attend"):
            image, draw = canvas(
                "KV cache: reuse projections, still read the history",
                "A causal full-attention example. Each tile stands for one token's K and V.",
                f"{'Prefill' if count == 3 else 'Decode'} | {count} tokens processed | {phase}",
            )
            text(draw, (32, 127), "Current prefix", 19, MUTED)
            for index in range(count):
                box(draw, (220 + index * 110, 120, 318 + index * 110, 169),
                    tokens[index], LIGHT_BLUE, BLUE)
            text(draw, (32, 216), "Without cache", 21)
            text(draw, (32, 319), "With cache", 21)
            for index in range(count):
                rect_top = (220 + index * 110, 202, 318 + index * 110, 260)
                rect_bottom = (220 + index * 110, 307, 318 + index * 110, 365)
                is_new = count == 3 or index == count - 1
                if phase == "project":
                    box(draw, rect_top, "compute", LIGHT_ORANGE, ORANGE, 17)
                    box(draw, rect_bottom, "compute" if is_new else "reuse",
                        LIGHT_ORANGE if is_new else LIGHT_TEAL, ORANGE if is_new else TEAL, 17)
                else:
                    box(draw, rect_top, "read", LIGHT_BLUE, BLUE, 18)
                    box(draw, rect_bottom, "read", LIGHT_BLUE, BLUE, 18)
            if phase == "project":
                new_count = count if count == 3 else 1
                text(draw, (220, 391),
                     f"K/V token rows this step: {count} without cache; {new_count} with cache.", 21)
                text(draw, (220, 437), "Prefill creates the initial cache; decode appends new state.", 19, MUTED)
            else:
                box(draw, (220, 395, 450, 443), "Current query q", LIGHT_BLUE, BLUE)
                arrow(draw, (455, 420), (565, 420))
                text(draw, (585, 403), "scores -> softmax", 21)
                text(draw, (585, 438), "-> weighted values", 21)
                text(draw, (220, 490), "Caching does not remove attention over earlier tokens.", 20, MUTED)
            frames.append(image)
    save("kv-cache", frames, [1500] * 7 + [2700], 4)


def quantization():
    frames = []
    values = [-0.87, -0.43, -0.12, 0.08, 0.37, 0.91]
    scale = 0.25
    def xp(value):
        return 500 + value * 350
    for step in range(7):
        fraction = step / 6
        image, draw = canvas(
            "Quantization: map real values to a finite grid",
            "Toy symmetric grid: scale = 0.25; codes -4 to 4; no clipping in this example.",
            f"Rounding transition {step}/6 | This illustrates the mapping, not kernel execution time.",
        )
        text(draw, (32, 128), "Original values", 21)
        text(draw, (32, 312), "Quantized reconstruction", 21)
        for y in (225, 408):
            draw.line((125, y, 875, y), fill=MUTED, width=2)
        for code in range(-4, 5):
            x = xp(code * scale)
            draw.line((x, 395, x, 421), fill=TEAL, width=2)
            text(draw, (x - 20, 436), f"{code * scale:g}", 17)
        for value in values:
            reconstructed = round(value / scale) * scale
            start, end = xp(value), xp(reconstructed)
            draw.ellipse((start - 6, 219, start + 6, 231), fill=BLUE)
            moving_x = start + fraction * (end - start)
            moving_y = 225 + fraction * (408 - 225)
            draw.line((start, 237, end, 394), fill=GREY, width=2)
            draw.ellipse((moving_x - 7, moving_y - 7, moving_x + 7, moving_y + 7), fill=ORANGE)
            text(draw, (start - 22, 185), f"{value:g}", 17, BLUE)
        text(draw, (180, 492), "0.37 / 0.25 -> code 1 -> reconstructed 0.25; error = -0.12", 21)
        frames.append(image)
    save("quantization-grid", frames, [450] * 6 + [3200], 6)


def batching():
    # All requests are available at t=0; prefill and variable iteration time are omitted.
    requests = [("A", 2), ("B", 5), ("C", 3), ("D", 2)]
    colors = {"A": LIGHT_BLUE, "B": LIGHT_TEAL, "C": LIGHT_ORANGE, "D": "#ede9fe"}
    fixed = [[("A", 1), ("B", 1)], [("A", 2), ("B", 2)],
             [None, ("B", 3)], [None, ("B", 4)], [None, ("B", 5)],
             [("C", 1), ("D", 1)], [("C", 2), ("D", 2)], [("C", 3), None]]
    continuous = [[("A", 1), ("B", 1)], [("A", 2), ("B", 2)],
                  [("C", 1), ("B", 3)], [("C", 2), ("B", 4)],
                  [("C", 3), ("B", 5)], [("D", 1), None],
                  [("D", 2), None], [None, None]]
    lengths = dict(requests)
    assert sum(item is not None for row in fixed for item in row) == sum(lengths.values())
    assert sum(item is not None for row in continuous for item in row) == sum(lengths.values())
    frames = []
    for tick in range(9):
        image, draw = canvas(
            "Continuous batching: reuse a slot when a request finishes",
            "Two slots; all requests ready initially. A: 2 tokens, B: 5, C: 3, D: 2.",
            f"After {tick} iterations | Schematic: equal decode ticks; admission/prefill costs omitted.",
        )
        for panel, schedule, top in (("Fixed batches", fixed, 166),
                                     ("Continuous batch", continuous, 345)):
            text(draw, (32, top + 20), panel, 19)
            for step in range(8):
                for slot in range(2):
                    left = 235 + step * 87
                    y = top + slot * 59
                    item = schedule[step][slot] if step < tick else None
                    if step >= tick:
                        label, fill = "", BG
                    elif item is None:
                        label, fill = "idle", GREY
                    else:
                        name, token = item
                        label, fill = f"{name}:{token}", colors[name]
                    box(draw, (left, y, left + 76, y + 47), label, fill, MUTED, 18)
                text(draw, (left + 25, top - 29), str(step + 1), 17, MUTED)
        text(draw, (238, 488), "C enters after A finishes instead of waiting for B.", 21, TEAL)
        frames.append(image)
    save("continuous-batching", frames, [850] * 8 + [3200], 8)


if __name__ == "__main__":
    gradient_descent()
    kv_cache()
    quantization()
    batching()
