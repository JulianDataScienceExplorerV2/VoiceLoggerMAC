#!/bin/bash
# Empaqueta la app nativa como "VoiceLogger Mac.app" (doble click, ícono en Finder/Dock,
# permisos de Recordatorios y micrófono). Uso, desde cualquier carpeta:
#   native/package.sh             → native/dist/VoiceLogger Mac.app
#   native/package.sh --dmg       → además native/dist/VoiceLoggerMac.dmg (para compartir)
#   native/package.sh --install   → además la copia a /Applications
#   SIGN_IDENTITY="Apple Development: …" native/package.sh   (si no, usa el primer certificado que
#                                     encuentre: Developer ID, Apple Development o «VoiceLogger»; si no hay, ad-hoc)
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
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "Opción desconocida: $arg (usa --help)" >&2; exit 1 ;;
    esac
done

# Primer certificado de firma de código disponible, como "HASH<TAB>NOMBRE".
# Con un certificado, macOS conserva los permisos (grabación, Recordatorios)
# al volver a empaquetar; con firma ad-hoc los olvida en cada build.
pick_identity() {
    local identities line pattern
    identities="$(security find-identity -p codesigning 2>/dev/null || true)"
    for pattern in "Developer ID Application" "Apple Development" "VoiceLogger"; do
        line="$(printf '%s\n' "$identities" | grep -m1 "\"$pattern" || true)"
        if [ -n "$line" ]; then
            printf '%s\t%s\n' "$(printf '%s' "$line" | awk '{print $2}')" "$(printf '%s' "$line" | sed -E 's/.*"(.*)".*/\1/')"
            return
        fi
    done
}

xml_escape() {
    printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'
}

cd "$NATIVE_DIR"
# xcodebuild (no «swift build»): es lo único que compila los shaders Metal de MLX
# (default.metallib), que necesita el motor de IA «Swift nativo». Requiere Xcode.
if ! xcodebuild -version >/dev/null 2>&1; then
    echo "Necesitas Xcode (no solo las Command Line Tools): instálalo desde la App Store y ejecuta" >&2
    echo "  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" >&2
    exit 1
fi
echo "→ Compilando en modo release con xcodebuild (la primera vez tarda: compila MLX)…"
DERIVED="$NATIVE_DIR/.build/xcode"
xcodebuild -scheme "$EXECUTABLE" -destination 'platform=macOS' -configuration Release \
    -derivedDataPath "$DERIVED" -skipMacroValidation -skipPackagePluginValidation -quiet build
BIN_DIR="$DERIVED/Build/Products/Release"

echo "→ Armando $APP_NAME.app…"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN_DIR/$EXECUTABLE" "$APP/Contents/MacOS/$EXECUTABLE"
# Recursos de SwiftPM del paquete y sus dependencias: entre ellos mlx-swift_Cmlx.bundle,
# con los shaders de MLX (default.metallib).
find "$BIN_DIR" -maxdepth 1 -name "*.bundle" -exec cp -R {} "$APP/Contents/Resources/" \;
if ! find "$APP/Contents/Resources" -name "*.metallib" | grep -q .; then
    echo "  (aviso: no encontré default.metallib; el motor de IA «Swift nativo» no funcionará, el de Python sí)"
fi
# Frameworks dinámicos de las dependencias, si los hay.
if [ -d "$BIN_DIR/PackageFrameworks" ]; then
    mkdir -p "$APP/Contents/Frameworks"
    cp -R "$BIN_DIR/PackageFrameworks/"*.framework "$APP/Contents/Frameworks/" 2>/dev/null || true
    install_name_tool -add_rpath "@executable_path/../Frameworks" "$APP/Contents/MacOS/$EXECUTABLE" 2>/dev/null || true
fi
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

SIGN_HASH="${SIGN_IDENTITY:-}"
SIGN_NAME="${SIGN_IDENTITY:-}"
if [ -z "$SIGN_HASH" ]; then
    PICKED="$(pick_identity)"
    if [ -n "$PICKED" ]; then
        SIGN_HASH="$(printf '%s' "$PICKED" | cut -f1)"
        SIGN_NAME="$(printf '%s' "$PICKED" | cut -f2)"
    fi
fi
if [ -n "$SIGN_HASH" ] && [ "$SIGN_HASH" != "-" ]; then
    echo "→ Firmando con el certificado: $SIGN_NAME"
    codesign --force --deep --identifier "$BUNDLE_ID" --sign "$SIGN_HASH" "$APP"
    STABLE_SIGNATURE=1
else
    echo "→ Firmando ad-hoc (sin certificado)…"
    codesign --force --deep --identifier "$BUNDLE_ID" --sign - "$APP"
    STABLE_SIGNATURE=0
fi

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

if [ "$STABLE_SIGNATURE" = 0 ]; then
    # El permiso guardado era para el build anterior y ya no vale, aunque Ajustes lo
    # siga mostrando activado. Se borra para que macOS lo pida limpio al abrir la app.
    tccutil reset All "$BUNDLE_ID" >/dev/null 2>&1 || true
    echo "  Permisos de $APP_NAME reiniciados: con firma ad-hoc macOS los olvida en cada empaquetado."
    echo "  Para que no vuelva a pasar, crea una vez un certificado «VoiceLogger» (ver native/README.md)."
fi

echo "✓ Listo: $APP"
echo "  Ábrela con doble click (o: open \"$APP\")."
echo "  La primera vez macOS pedirá permiso de Grabación de pantalla y audio del sistema."
