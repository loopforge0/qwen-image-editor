#!/bin/bash
# Build the Mac app and disk image: dist/QwenImageEditor-<version>.dmg. Run on a Mac (needs sips, iconutil, hdiutil).
# Nothing of Qwen's is included: the editor's setup page downloads and converts the model on the user's Mac.
#
#   bash installer/macos/build.sh       (UV_VERSION=... to use a different uv; "bash" avoids macOS refusing to run
#                                        a script from a downloaded zip: "operation not permitted")
set -euo pipefail

UV_VERSION="${UV_VERSION:-0.11.28}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)"
BUILD="$HERE/build"
APP="$BUILD/Qwen Image Editor.app"

rm -rf "$BUILD"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/bin" "$APP/Contents/Resources/app"

# The editor: what a user needs to run it, not the development and test material.
for item in engine app pyproject.toml uv.lock README.md LICENSE; do
  cp -R "$ROOT/$item" "$APP/Contents/Resources/app/"
done
find "$APP" -name "__pycache__" -type d -prune -exec rm -rf {} +

tr -d '\r' < "$HERE/launcher.sh" > "$APP/Contents/MacOS/Qwen Image Editor"  # LF only, even if edited on Windows
chmod +x "$APP/Contents/MacOS/Qwen Image Editor"
tr -d '\r' < "$HERE/progress.js" > "$APP/Contents/Resources/progress.js"  # the first-start progress window
sed "s/@VERSION@/$VERSION/g" "$HERE/Info.plist" | tr -d '\r' > "$APP/Contents/Info.plist"

# uv: installs Python and the editor's packages on first start.
echo "Downloading uv $UV_VERSION"
curl -fsSL "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-aarch64-apple-darwin.tar.gz" | tar -xz -C "$BUILD"
cp "$BUILD/uv-aarch64-apple-darwin/uv" "$APP/Contents/Resources/bin/uv"

# Icon
ICONSET="$BUILD/AppIcon.iconset"
mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
  sips -z $s $s "$ROOT/installer/assets/icon-1024.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  sips -z $((s * 2)) $((s * 2)) "$ROOT/installer/assets/icon-1024.png" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"

# Nothing about the build machine goes in: drop extended attributes (download origins, quarantine, provenance).
xattr -cr "$APP"

# Ad-hoc signature. Without an Apple Developer ID, macOS still asks the user to allow the app the first time
# (System Settings > Privacy & Security > Open Anyway).
codesign --force --deep --sign - "$APP"
# The same fixed date on every file (after signing, which writes files too), so the app doesn't record when it was
# built. Timestamps aren't part of the signature.
find "$APP" -exec touch -h -t 202601010000 {} +

# Disk image with the usual "drag to Applications" layout.
DMG_DIR="$BUILD/dmg"
mkdir -p "$DMG_DIR"
ditto --noextattr --noqtn "$APP" "$DMG_DIR/Qwen Image Editor.app"  # keeps the fixed dates, no attributes
ln -s /Applications "$DMG_DIR/Applications"
mkdir -p "$ROOT/dist"
hdiutil create -volname "Qwen Image Editor" -srcfolder "$DMG_DIR" -ov -format UDZO "$ROOT/dist/QwenImageEditor-$VERSION.dmg" >/dev/null
echo "Built $ROOT/dist/QwenImageEditor-$VERSION.dmg"
