import EventKit
import Foundation

/// An action item extracted from the meeting by Qwen.
struct MeetingTask: Identifiable, Decodable, Sendable {
    var id = UUID()
    let title: String
    /// Person responsible, or "" if nobody was mentioned.
    let owner: String
    /// Deadline as YYYY-MM-DD, or "".
    let due: String
    /// Checked in the "add to Reminders" sheet.
    var selected = true

    enum CodingKeys: String, CodingKey {
        case title, owner, due
    }

    var dueComponents: DateComponents? {
        let parts = due.split(separator: "-").compactMap { Int($0) }
        guard parts.count == 3 else { return nil }
        return DateComponents(year: parts[0], month: parts[1], day: parts[2])
    }

    /// "- [ ] Title — Owner (2026-11-13)", for Markdown and the clipboard.
    var markdownLine: String {
        var line = "- [ ] \(title)"
        if !owner.isEmpty { line += " — \(owner)" }
        if !due.isEmpty { line += " (\(due))" }
        return line
    }
}

/// Adds meeting tasks to the Reminders app (EventKit).
@MainActor
final class RemindersExporter {
    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    private let store = EKEventStore()

    /// macOS kills an app that asks for Reminders access without a usage
    /// description in its Info.plist. `swift run` has no Info.plist, so the
    /// feature is only offered in the packaged app (native/package.sh).
    nonisolated static var isAvailable: Bool {
        Bundle.main.object(forInfoDictionaryKey: "NSRemindersFullAccessUsageDescription") != nil
    }

    /// Saves the tasks in the list `listName` (created if missing; falls back
    /// to the default list). Returns how many were added.
    func add(_ tasks: [MeetingTask], listName: String, meetingTitle: String) async throws -> Int {
        guard Self.isAvailable else {
            throw Failure(message: "Recordatorios necesita la app empaquetada: ejecuta native/package.sh y ábrela desde Finder.")
        }
        guard try await requestAccess() else {
            throw Failure(message: "Sin permiso para Recordatorios. Actívalo en Ajustes del Sistema › Privacidad y seguridad › Recordatorios.")
        }
        guard let calendar = list(named: listName) ?? store.defaultCalendarForNewReminders() else {
            throw Failure(message: "No encontré ninguna lista de Recordatorios.")
        }
        for task in tasks {
            let reminder = EKReminder(eventStore: store)
            reminder.title = task.title
            reminder.calendar = calendar
            reminder.notes = task.owner.isEmpty ? meetingTitle : "Responsable: \(task.owner)\n\(meetingTitle)"
            if let due = task.dueComponents {
                reminder.dueDateComponents = due
            }
            try store.save(reminder, commit: false)
        }
        try store.commit()
        return tasks.count
    }

    private func requestAccess() async throws -> Bool {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Bool, Error>) in
            // @Sendable: EventKit calls this on a background queue.
            store.requestFullAccessToReminders { @Sendable granted, error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: granted)
                }
            }
        }
    }

    private func list(named name: String) -> EKCalendar? {
        if let existing = store.calendars(for: .reminder).first(where: { $0.title == name }) {
            return existing
        }
        guard let source = store.defaultCalendarForNewReminders()?.source else { return nil }
        let calendar = EKCalendar(for: .reminder, eventStore: store)
        calendar.title = name
        calendar.source = source
        do {
            try store.saveCalendar(calendar, commit: true)
            return calendar
        } catch {
            return nil
        }
    }
}
