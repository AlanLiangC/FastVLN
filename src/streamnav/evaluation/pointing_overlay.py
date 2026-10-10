"""High-contrast markers for frame-relative navigation point predictions."""

from functools import lru_cache

from PIL import ImageDraw, ImageFont

from streamnav.contracts.perception import decode_pixel

APOS_COLOR = "#39ff14"
OPOS_COLOR = "#ff30e8"


@lru_cache(maxsize=8)
def _font(size):
    return ImageFont.load_default(size=size)


def draw_pointing_overlay(canvas, perception, image_size):
    """Draw only point tokens, preserving their exact decoded image positions."""
    width, height = image_size
    # Crop the drawing surface so edge markers never spill onto the status panel.
    image = canvas.crop((0, 0, width, height))
    draw = ImageDraw.Draw(image)
    radius = max(8, round(width / 48))
    font = _font(max(12, round(width / 34)))
    labels = []
    for name, color in (("apos", APOS_COLOR), ("opos", OPOS_COLOR)):
        point = decode_pixel(perception[name], width, height)
        if point is None:
            continue
        x, y = point
        if name == "apos":
            for inset, fill in ((0, "black"), (2, "white"), (4, color)):
                r = radius - inset
                draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)
            draw.line((x - 3, y, x + 3, y), fill="black", width=1)
            draw.line((x, y - 3, x, y + 3), fill="black", width=1)
        else:
            # A larger hollow diamond keeps both markers visible if points coincide.
            r = radius + 4
            vertices = [(x, y - r), (x + r, y), (x, y + r), (x - r, y), (x, y - r)]
            for outline, stroke in (("black", 7), ("white", 5), (color, 3)):
                draw.line(vertices, fill=outline, width=stroke, joint="curve")
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline="black")
        labels.append((name, color, x, y))

    # Draw labels after all markers; APOS above / OPOS below also separates overlap.
    for name, color, x, y in labels:
        label = name.upper()
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        box_w, box_h = right - left + 8, bottom - top + 8
        label_x = x + radius + 9
        if label_x + box_w > width - 2:
            label_x = x - radius - 9 - box_w
        label_y = y - box_h - 4 if name == "apos" else y + 4
        label_x = max(2, min(label_x, width - box_w - 2))
        label_y = max(2, min(label_y, height - box_h - 2))
        draw.rectangle(
            (label_x, label_y, label_x + box_w, label_y + box_h),
            fill="black",
            outline=color,
            width=2,
        )
        draw.text((label_x + 4 - left, label_y + 4 - top), label, fill=color, font=font)
    canvas.paste(image, (0, 0))
