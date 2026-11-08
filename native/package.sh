#!/bin/bash
# Empaqueta la app nativa como "VoiceLogger Mac.app" (doble click, ícono en Finder/Dock,
# permisos de Recordatorios y micrófono). Uso, desde cualquier carpeta:
#   native/package.sh             → native/dist/VoiceLogger Mac.app
#   native/package.sh --dmg       → además native/dist/VoiceLoggerMac.dmg (para compartir)
#   native/package.sh --install   → además la copia a /Applications
#   SIGN_IDENTITY="Developer ID Application: …" native/package.sh   (por defecto: firma ad-hoc)
# La IA (Qwen) sigue usando el .venv de este repo: la ruta del repo queda guardada en la app.
set -euo pipefail

NATIVE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$NATIVE_DIR/.." && pwd)"
APP_NAME="VoiceLogger Mac"
EXECUTABLE="VoiceLoggerMacNative"
BUNDLE_ID="com.julian.voiceloggermac.native"
VERSION="${VERSION:-1.0}"
DIST="$NATIVE_DIR/dist"
APP="$DIST/$APP_NAME.app"
ICON="$REPO_ROOT/VoiceLoggerMac.app/Contents/Resources/AppIcon.icns"

MAKE_DMG=0
INSTALL=0
for arg in "$@"; do
    case "$arg" in
        --dmg) MAKE_DMG=1 ;;
        --install) INSTALL=1 ;;
        -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
        *) echo "Opción desconocida: $arg (usa --help)" >&2; exit 1 ;;
    esac
done

xml_escape() {
    printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'
}

cd "$NATIVE_DIR"
echo "→ Compilando en modo release…"
swift build -c release --product "$EXECUTABLE"
BIN_DIR="$(swift build -c release --show-bin-path)"

echo "→ Armando $APP_NAME.app…"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN_DIR/$EXECUTABLE" "$APP/Contents/MacOS/$EXECUTABLE"
# Recursos de SwiftPM del paquete y sus dependencias (si los hay).
find "$BIN_DIR" -maxdepth 1 -name "*.bundle" -exec cp -R {} "$APP/Contents/Resources/" \;
if [ -f "$ICON" ]; then
    cp "$ICON" "$APP/Contents/Resources/AppIcon.icns"
else
    echo "  (aviso: no encontré $ICON; la app usará el ícono dibujado en código)"
fi

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundleDisplayName</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleExecutable</key>
    <string>$EXECUTABLE</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>15.0</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.productivity</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>VoiceLogger Mac usa el micrófono solo si eliges esa fuente de audio.</string>
    <key>NSRemindersFullAccessUsageDescription</key>
    <string>VoiceLogger Mac agrega a Recordatorios las tareas detectadas en tus reuniones, solo cuando tú lo confirmas.</string>
    <key>NSRemindersUsageDescription</key>
    <string>VoiceLogger Mac agrega a Recordatorios las tareas detectadas en tus reuniones, solo cuando tú lo confirmas.</string>
    <key>VoiceLoggerRepoRoot</key>
    <string>$(xml_escape "$REPO_ROOT")</string>
</dict>
</plist>
PLIST
plutil -lint "$APP/Contents/Info.plist" >/dev/null

echo "→ Firmando (${SIGN_IDENTITY:-ad-hoc})…"
codesign --force --deep --sign "${SIGN_IDENTITY:--}" "$APP"

if [ "$MAKE_DMG" = 1 ]; then
    echo "→ Creando DMG…"
    STAGING="$(mktemp -d)"
    trap 'rm -rf "$STAGING"' EXIT
    cp -R "$APP" "$STAGING/"
    ln -s /Applications "$STAGING/Applications"
    rm -f "$DIST/VoiceLoggerMac.dmg"
    hdiutil create -volname "$APP_NAME" -srcfolder "$STAGING" -ov -format UDZO "$DIST/VoiceLoggerMac.dmg" >/dev/null
    echo "  DMG: $DIST/VoiceLoggerMac.dmg"
fi

if [ "$INSTALL" = 1 ]; then
    echo "→ Instalando en /Applications…"
    rm -rf "/Applications/$APP_NAME.app"
    ditto "$APP" "/Applications/$APP_NAME.app"
    echo "  Instalada: /Applications/$APP_NAME.app"
fi

echo "✓ Listo: $APP"
echo "  Ábrela con doble click (o: open \"$APP\")."
echo "  La primera vez macOS pedirá permiso de Grabación de pantalla y audio del sistema."
