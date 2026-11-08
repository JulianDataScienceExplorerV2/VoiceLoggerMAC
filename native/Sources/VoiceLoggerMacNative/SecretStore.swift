import Foundation
import Security

/// Small secrets in the macOS Keychain (login keychain), e.g. the Tavily key:
/// never in UserDefaults, the repo or the logs.
enum SecretStore {
    static let service = "com.julian.voiceloggermac.native"

    private static func baseQuery(_ account: String) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
    }

    static func read(_ account: String) -> String? {
        var query = baseQuery(account)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess, let data = item as? Data else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }

    @discardableResult
    static func save(_ value: String, account: String) -> Bool {
        delete(account)
        var attributes = baseQuery(account)
        attributes[kSecValueData as String] = Data(value.utf8)
        attributes[kSecAttrLabel as String] = "VoiceLogger Mac · \(account)"
        return SecItemAdd(attributes as CFDictionary, nil) == errSecSuccess
    }

    static func delete(_ account: String) {
        SecItemDelete(baseQuery(account) as CFDictionary)
    }
}

/// The Tavily key, read from the Keychain the first time a search needs it
/// (not at launch: with an ad-hoc signed build macOS may ask once to allow it).
final class TavilyKey: @unchecked Sendable {
    static let shared = TavilyKey()
    static let account = "tavily"

    private let lock = NSLock()
    private var value: String?
    private var loaded = false

    var current: String? {
        lock.lock()
        defer { lock.unlock() }
        if !loaded {
            value = SecretStore.read(Self.account)
            loaded = true
        }
        return value
    }

    func set(_ newValue: String?) {
        lock.lock()
        defer { lock.unlock() }
        value = newValue
        loaded = true
    }
}
