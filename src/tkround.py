"""Rounded ttk buttons.

ttk has no border-radius: clam draws a button as a rectangular bevel and no amount of style
configuration rounds it off. The usual workaround is to abandon ttk.Button for a tk.Canvas widget,
which loses keyboard traversal, the disabled/pressed/active/selected state machine, focus rings and
the named font, and then has to reimplement all of it badly.

This module keeps the real ttk.Button and only replaces what it draws: a PIL-rendered rounded
rectangle registered as a themed *image element*, 9-slice scaled (`border=`) so one small bitmap
stretches to any button width without distorting the corners. The element carries a state map, so
hover / pressed / disabled / selected / focused each get their own bitmap and the widget keeps
behaving like a button.

Three things to know when using it:

* The corners are drawn OPAQUE in the parent's background colour, not with an alpha channel. Tk
  composites an RGBA image element against whatever happens to be beneath it, which is not something
  worth depending on; painting the known parent colour into the corners is deterministic. So
  `behind` must be the background the button actually sits on -- pass the wrong colour and you get
  four visible corner squares.
* State order is significant: ttk uses the FIRST spec that matches. A selected chip that is also
  hovered therefore shows its selected bitmap, because "selected" is listed before "active".
* Element names are global to the Tcl interpreter and `element_create` fails if the name already
  exists. Dashboard re-runs its whole styling pass on every resize (fonts and paddings are baked into
  the styles), so the registry below hands back the existing element instead of recreating it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from tkinter import TclError, ttk

from PIL import Image, ImageDraw, ImageTk

# The corner arc is drawn at 4x and downsampled: a directly-rasterised 7 px radius is visibly
# stair-stepped, and the bitmap is small enough that the extra work is invisible.
SUPERSAMPLE = 4


@dataclass(frozen=True)
class Tier:
    """One button tier: the resting fill/ring plus a bitmap per widget state.

    `edge` equal to `fill` gives a solid button with no visible ring, which is what the filled
    primary wants; a quiet tier sets `fill` to the parent colour so only the label marks it out until
    the pointer arrives.
    """
    fill: str
    edge: str
    # (state, fill, edge), most specific first -- see the module docstring on ordering.
    states: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)


def _rounded(size: int, radius: int, behind: str, fill: str, edge: str) -> Image.Image:
    ss = SUPERSAMPLE
    big = size * ss
    img = Image.new("RGB", (big, big), behind)
    draw = ImageDraw.Draw(img)
    # inset by half the line width so the ring lands inside the bitmap instead of being clipped by
    # the right/bottom edge
    draw.rounded_rectangle(
        (ss // 2, ss // 2, big - 1 - ss // 2, big - 1 - ss // 2),
        radius=radius * ss, fill=fill, outline=edge, width=ss,
    )
    return img.resize((size, size), Image.LANCZOS)


class RoundedStyles:
    """Creates (and memoises) rounded ttk button styles on one ttk.Style."""

    def __init__(self, style: ttk.Style) -> None:
        self._style = style
        # PhotoImages are referenced only by Tcl once handed to element_create; without a Python
        # reference they are garbage collected and the buttons render as empty holes.
        self._images: list[ImageTk.PhotoImage] = []
        self._elements: dict[tuple, str] = {}

    def element(self, key: str, tier: Tier, *, radius: int, behind: str) -> str:
        """Name of the image element for this tier, creating it on first use."""
        cache_key = (key, radius, behind, tier.fill, tier.edge, tier.states)
        if cache_key in self._elements:
            return self._elements[cache_key]

        # border must exceed radius or the 9-slice stretch would cut into the arc; the +2 keeps the
        # ring itself inside the fixed border area. size leaves a 1 px stretchable middle.
        border = radius + 2
        size = 2 * border + 1

        def photo(fill: str, edge: str) -> ImageTk.PhotoImage:
            image = ImageTk.PhotoImage(_rounded(size, radius, behind, fill, edge))
            self._images.append(image)     # keep alive
            return image

        args: list = [photo(tier.fill, tier.edge)]
        for state, fill, edge in tier.states:
            args.append((state, photo(fill, edge)))

        name = f"SRRound{len(self._elements)}.{key}.button"
        try:
            self._style.element_create(name, "image", *args, border=border, sticky="nsew")
        except TclError:
            # Only reachable if a second Dashboard shares the interpreter (tests); suffixing keeps
            # both alive rather than crashing the newer one.
            name = f"{name}.{id(self)}"
            self._style.element_create(name, "image", *args, border=border, sticky="nsew")
        self._elements[cache_key] = name
        return name

    def button_style(self, name: str, tier: Tier, *, radius: int, behind: str,
                     foreground: str, foreground_hover: str, foreground_disabled: str,
                     font: str, padding: tuple[int, int],
                     element_prefix: str = "Button",
                     foreground_selected: str | None = None) -> None:
        """Define (or redefine after a rescale) `name` as a rounded button style.

        element_prefix must match the widget class whose layout is being replaced -- "Button" for
        ttk.Button, "Toolbutton" for a Radiobutton/Checkbutton drawn as a chip. ttk resolves
        "<Prefix>.padding" down to the generic padding element, so only the layout name changes.
        """
        element = self.element(f"{name}~{element_prefix}", tier, radius=radius, behind=behind)
        self._style.layout(name, [
            (element, {"sticky": "nsew", "children": [
                (f"{element_prefix}.padding", {"sticky": "nsew", "children": [
                    (f"{element_prefix}.label", {"sticky": "nsew"}),
                ]}),
            ]}),
        ])
        self._style.configure(name, font=font, padding=padding, anchor="center",
                              foreground=foreground, background=behind,
                              borderwidth=0, relief="flat", focusthickness=0)
        mapping = [("disabled", foreground_disabled)]
        if foreground_selected is not None:
            mapping.append(("selected", foreground_selected))
        mapping.append(("active", foreground_hover))
        self._style.map(name, foreground=mapping)
