import AppKit
import SwiftUI

/// App icon drawn in SwiftUI, same design as `assets/AppIcon.svg`: voice
/// bars (green → blue, like the transcript and questions) and the purple AI
/// sparkle. Rendered at launch so the Dock shows it even when the app runs
/// with `swift run` (no .app bundle to hold an .icns).
struct AppIconView: View {
    /// Bar heights on the 1024-point canvas, left to right.
    private static let bars: [CGFloat] = [156, 302, 448, 270, 146]

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 185, style: .continuous)
                .fill(LinearGradient(
                    colors: [Color(red: 0.137, green: 0.137, blue: 0.227), Color(red: 0.043, green: 0.043, blue: 0.071)],
                    startPoint: .top, endPoint: .bottom
                ))
                .overlay {
                    RoundedRectangle(cornerRadius: 182, style: .continuous)
                        .strokeBorder(Color.white.opacity(0.09), lineWidth: 6)
                }
                .frame(width: 824, height: 824)
                .shadow(color: .black.opacity(0.45), radius: 18, y: 14)

            HStack(spacing: 44) {
                ForEach(Array(Self.bars.enumerated()), id: \.offset) { item in
                    Capsule()
                        .fill(LinearGradient(colors: [Theme.brand, Theme.qa], startPoint: .bottom, endPoint: .top))
                        .frame(width: 70, height: item.element)
                }
            }
            .shadow(color: Theme.qa.opacity(0.5), radius: 14)
            .offset(y: 28)

            Group {
                Sparkle().frame(width: 190, height: 190).position(x: 735, y: 300)
                Sparkle().frame(width: 80, height: 80).position(x: 826, y: 410)
            }
            .foregroundStyle(LinearGradient(
                colors: [Color(red: 0.84, green: 0.72, blue: 1.0), Theme.ai],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
            .shadow(color: Theme.ai.opacity(0.6), radius: 14)
        }
        .frame(width: 1024, height: 1024)
    }
}

/// Four-point star with curved sides.
struct Sparkle: Shape {
    func path(in rect: CGRect) -> Path {
        let c = CGPoint(x: rect.midX, y: rect.midY)
        var path = Path()
        path.move(to: CGPoint(x: c.x, y: rect.minY))
        path.addQuadCurve(to: CGPoint(x: rect.maxX, y: c.y), control: c)
        path.addQuadCurve(to: CGPoint(x: c.x, y: rect.maxY), control: c)
        path.addQuadCurve(to: CGPoint(x: rect.minX, y: c.y), control: c)
        path.addQuadCurve(to: CGPoint(x: c.x, y: rect.minY), control: c)
        path.closeSubpath()
        return path
    }
}

@MainActor
enum AppIcon {
    /// Sets the Dock / app switcher icon.
    static func apply() {
        let renderer = ImageRenderer(content: AppIconView())
        renderer.scale = 1
        if let image = renderer.nsImage {
            NSApp.applicationIconImage = image
        }
    }
}
