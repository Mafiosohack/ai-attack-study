#!/usr/bin/env python3
"""Generate the app icon. Committed output: victim_chat.ico (regenerate with
`python tools/victim_chat/make_icon.py`). Mirrors the empty-state badge: emerald square,
white ring, emerald dot.
"""
from pathlib import Path
from PIL import Image, ImageDraw

SIZE = 512
EMERALD, PALE, WHITE = (4, 120, 87), (167, 243, 208), (255, 255, 255)

img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle((0, 0, SIZE - 1, SIZE - 1), radius=int(SIZE * 0.22), fill=EMERALD)
# Speech-bubble ring with a solid centre — reads at 16px as a green dot.
d.ellipse((SIZE * 0.22, SIZE * 0.22, SIZE * 0.78, SIZE * 0.78), outline=PALE,
          width=int(SIZE * 0.045))
d.ellipse((SIZE * 0.40, SIZE * 0.40, SIZE * 0.60, SIZE * 0.60), fill=WHITE)
# Tail, so the mark reads as conversation rather than a generic badge.
d.polygon([(SIZE * 0.34, SIZE * 0.70), (SIZE * 0.46, SIZE * 0.78),
           (SIZE * 0.30, SIZE * 0.86)], fill=PALE)

out = Path(__file__).resolve().parent / "victim_chat.ico"
img.save(out, format="ICO",
         sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("wrote", out)
