"""Draw the app icon (the same mark as the editor's logo) as icon.ico (Windows) and icon-1024.png (macOS builds it
into an .icns). Run once after changing the design:  uv run python installer/assets/make_icons.py"""

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
S = 1024  # drawn at 4x and scaled down for smooth edges


def draw(size=S * 4) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    u = size / 32  # the editor's logo is drawn on a 32-unit grid
    d.rounded_rectangle([3 * u, 3 * u, 29 * u, 29 * u], radius=7 * u, fill=(236, 236, 238, 255))
    d.ellipse([16.5 * u, 8.5 * u, 23.5 * u, 15.5 * u], fill=(17, 17, 19, 255))
    d.polygon([(7 * u, 24 * u), (14 * u, 16 * u), (19 * u, 21 * u), (22 * u, 18 * u), (26 * u, 24 * u)], fill=(17, 17, 19, 255))
    return img


def main():
    big = draw().resize((S, S), Image.LANCZOS)
    big.save(HERE / "icon-1024.png")
    big.save(HERE / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("wrote", HERE / "icon-1024.png", "and", HERE / "icon.ico")


if __name__ == "__main__":
    main()
