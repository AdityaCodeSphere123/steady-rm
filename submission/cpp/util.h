#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace rm {

constexpr int MAX_THREADS = 4;

inline uint64_t hash_str(std::string_view s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    h ^= h >> 31; h *= 0x7fb5d329728ea185ull; h ^= h >> 27;
    return h;
}

class StrTable {
public:
    static constexpr uint32_t NONE = 0xFFFFFFFFu;

    explicit StrTable(size_t expected = 1024) { rehash(pow2_at_least(expected * 2)); }

    size_t size() const { return off_.size(); }

    std::string_view key(uint32_t id) const {
        return std::string_view(arena_.data() + off_[id], len_[id]);
    }

    uint32_t find(std::string_view s) const { return find_h(s, hash_str(s)); }

    uint32_t find_h(std::string_view s, uint64_t h) const {
        size_t i = h & mask_;
        for (;;) {
            uint32_t v = slots_[i];
            if (v == 0) return NONE;
            uint32_t id = v - 1;
            if (hash_[id] == h && eq(id, s)) return id;
            i = (i + 1) & mask_;
        }
    }

    uint32_t insert(std::string_view s, bool* inserted = nullptr) {
        return insert_h(s, hash_str(s), inserted);
    }

    uint32_t insert_h(std::string_view s, uint64_t h, bool* inserted = nullptr) {
        if ((off_.size() + 1) * 2 > slots_.size()) rehash(slots_.size() * 2);
        size_t i = h & mask_;
        for (;;) {
            uint32_t v = slots_[i];
            if (v == 0) break;
            uint32_t id = v - 1;
            if (hash_[id] == h && eq(id, s)) { if (inserted) *inserted = false; return id; }
            i = (i + 1) & mask_;
        }
        uint32_t id = static_cast<uint32_t>(off_.size());
        off_.push_back(arena_.size());
        len_.push_back(static_cast<uint32_t>(s.size()));
        hash_.push_back(h);
        arena_.insert(arena_.end(), s.begin(), s.end());
        slots_[i] = id + 1;
        if (inserted) *inserted = true;
        return id;
    }

    uint64_t hash_of(uint32_t id) const { return hash_[id]; }

private:
    static size_t pow2_at_least(size_t n) { size_t p = 16; while (p < n) p <<= 1; return p; }

    bool eq(uint32_t id, std::string_view s) const {
        return len_[id] == s.size() && std::memcmp(arena_.data() + off_[id], s.data(), s.size()) == 0;
    }

    void rehash(size_t n) {
        slots_.assign(n, 0);
        mask_ = n - 1;
        for (uint32_t id = 0; id < off_.size(); ++id) {
            size_t i = hash_[id] & mask_;
            while (slots_[i]) i = (i + 1) & mask_;
            slots_[i] = id + 1;
        }
    }

    std::vector<char> arena_;
    std::vector<uint64_t> off_;
    std::vector<uint32_t> len_;
    std::vector<uint64_t> hash_;
    std::vector<uint32_t> slots_;
    size_t mask_ = 0;
};

class ThreadPool {
public:
    explicit ThreadPool(int w) : W_(w < 1 ? 1 : (w > MAX_THREADS ? MAX_THREADS : w)) {
        for (int i = 1; i < W_; ++i) threads_.emplace_back([this, i] { loop(i); });
    }
    ~ThreadPool() {
        { std::lock_guard<std::mutex> g(m_); stop_ = true; ++gen_; }
        cv_.notify_all();
        for (auto& t : threads_) t.join();
    }
    int workers() const { return W_; }

    void run(size_t n, const std::function<void(size_t, int)>& fn) {
        if (n == 0) return;
        if (W_ == 1 || n == 1) { for (size_t i = 0; i < n; ++i) fn(i, 0); return; }
        std::lock_guard<std::mutex> job_guard(job_m_);
        {
            std::lock_guard<std::mutex> g(m_);
            fn_ = &fn; n_ = n; next_.store(0); pending_ = W_ - 1; ++gen_;
        }
        cv_.notify_all();
        work(0);
        std::unique_lock<std::mutex> lk(m_);
        done_cv_.wait(lk, [this] { return pending_ == 0; });
        fn_ = nullptr;
    }

private:
    void work(int w) {
        for (;;) {
            size_t i = next_.fetch_add(1);
            if (i >= n_) break;
            (*fn_)(i, w);
        }
    }
    void loop(int w) {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return gen_ != seen; });
                seen = gen_;
                if (stop_) return;
            }
            work(w);
            {
                std::lock_guard<std::mutex> g(m_);
                if (--pending_ == 0) done_cv_.notify_one();
            }
        }
    }

    int W_;
    std::vector<std::thread> threads_;
    std::mutex m_, job_m_;
    std::condition_variable cv_, done_cv_;
    const std::function<void(size_t, int)>* fn_ = nullptr;
    size_t n_ = 0;
    std::atomic<size_t> next_{0};
    int pending_ = 0;
    uint64_t gen_ = 0;
    bool stop_ = false;
};

inline void put_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) out += static_cast<char>(cp);
    else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18)); out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

inline int hexv(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

inline bool json_string(const char*& p, const char* end, std::string& buf, std::string_view& view) {
    const char* s = p;
    while (p < end && *p != '"' && *p != '\\') ++p;
    if (p >= end) return false;
    if (*p == '"') { view = std::string_view(s, p - s); ++p; return true; }
    buf.assign(s, p - s);
    while (p < end) {
        char c = *p++;
        if (c == '"') { view = buf; return true; }
        if (c != '\\') { buf += c; continue; }
        if (p >= end) return false;
        char e = *p++;
        switch (e) {
            case 'n': buf += '\n'; break;
            case 't': buf += '\t'; break;
            case 'r': buf += '\r'; break;
            case 'b': buf += '\b'; break;
            case 'f': buf += '\f'; break;
            case 'u': {
                if (end - p < 4) return false;
                uint32_t cp = 0;
                for (int i = 0; i < 4; ++i) { int h = hexv(p[i]); if (h < 0) return false; cp = cp * 16 + h; }
                p += 4;
                if (cp >= 0xD800 && cp < 0xDC00 && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                    uint32_t lo = 0; bool ok = true;
                    for (int i = 0; i < 4; ++i) { int h = hexv(p[2 + i]); if (h < 0) { ok = false; break; } lo = lo * 16 + h; }
                    if (ok && lo >= 0xDC00 && lo < 0xE000) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); p += 6; }
                }
                put_utf8(buf, cp);
                break;
            }
            default: buf += e; break;
        }
    }
    return false;
}

inline void skip_ws(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
}

inline bool json_skip(const char*& p, const char* end) {
    skip_ws(p, end);
    if (p >= end) return false;
    if (*p == '"') {
        ++p;
        while (p < end) { if (*p == '\\') p += 2; else if (*p == '"') { ++p; return true; } else ++p; }
        return false;
    }
    if (*p == '{' || *p == '[') {
        int depth = 0;
        while (p < end) {
            char c = *p;
            if (c == '"') { if (!json_skip(p, end)) return false; continue; }
            if (c == '{' || c == '[') ++depth;
            else if (c == '}' || c == ']') { --depth; if (depth == 0) { ++p; return true; } }
            ++p;
        }
        return false;
    }
    while (p < end && *p != ',' && *p != '}') ++p;
    return true;
}

struct JsonDoc {
    std::string id_buf, text_buf;
    std::string_view id, text;
    bool has_id = false;
};

inline bool parse_doc_line(const char* p, const char* end, JsonDoc& d) {
    d.has_id = false;
    d.text = std::string_view();
    skip_ws(p, end);
    if (p >= end || *p != '{') return false;
    ++p;
    std::string key_buf;
    for (;;) {
        skip_ws(p, end);
        if (p >= end) return d.has_id;
        if (*p == '}') return d.has_id;
        if (*p == ',') { ++p; continue; }
        if (*p != '"') return d.has_id;
        ++p;
        std::string_view key;
        if (!json_string(p, end, key_buf, key)) return d.has_id;
        skip_ws(p, end);
        if (p >= end || *p != ':') return d.has_id;
        ++p;
        skip_ws(p, end);
        if (key == "doc_id") {
            if (p < end && *p == '"') {
                ++p;
                if (!json_string(p, end, d.id_buf, d.id)) return false;
                if (d.id.data() != d.id_buf.data()) { d.id_buf.assign(d.id); d.id = d.id_buf; }
                d.has_id = true;
            } else {
                const char* s = p;
                while (p < end && *p != ',' && *p != '}' && *p != ' ') ++p;
                d.id_buf.assign(s, p - s); d.id = d.id_buf; d.has_id = true;
            }
        } else if (key == "text") {
            if (p < end && *p == '"') {
                ++p;
                if (!json_string(p, end, d.text_buf, d.text)) return false;
            } else if (!json_skip(p, end)) return false;
        } else if (!json_skip(p, end)) {
            return d.has_id;
        }
    }
}

}  // namespace rm
