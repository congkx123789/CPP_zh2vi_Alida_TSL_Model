#include "dictionary.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <unordered_set>

static std::vector<std::string> utf8_to_chars(const std::string& str) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < str.length()) {
        unsigned char c = (unsigned char)str[i];
        size_t len = 1;
        if ((c & 0x80) == 0) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        
        if (i + len <= str.length()) {
            chars.push_back(str.substr(i, len));
        } else {
            chars.push_back(str.substr(i));
        }
        i += len;
    }
    return chars;
}

VietphraseTrie::VietphraseTrie() : max_key_len(30), is_loaded(false) {}
VietphraseTrie::~VietphraseTrie() {}

bool VietphraseTrie::load(const std::string& data_dir) {
    std::string trie_path = data_dir + "/trie.marisa";
    std::string flat_path = data_dir + "/meanings_flat.bin";
    std::string offsets_path = data_dir + "/offsets.bin";
    std::string tiers_path = data_dir + "/tiers.bin";

    try {
        trie.mmap(trie_path.c_str());

        std::ifstream f_flat(flat_path, std::ios::binary | std::ios::ate);
        if (!f_flat.is_open()) return false;
        std::streamsize sz_flat = f_flat.tellg();
        f_flat.seekg(0, std::ios::beg);
        flat_bytes.resize(sz_flat);
        f_flat.read(flat_bytes.data(), sz_flat);

        std::ifstream f_off(offsets_path, std::ios::binary | std::ios::ate);
        if (!f_off.is_open()) return false;
        std::streamsize sz_off = f_off.tellg();
        f_off.seekg(0, std::ios::beg);
        offsets.resize(sz_off / sizeof(uint32_t));
        f_off.read((char*)offsets.data(), sz_off);

        std::ifstream f_tier(tiers_path, std::ios::binary | std::ios::ate);
        if (!f_tier.is_open()) return false;
        std::streamsize sz_tier = f_tier.tellg();
        f_tier.seekg(0, std::ios::beg);
        tiers.resize(sz_tier);
        f_tier.read((char*)tiers.data(), sz_tier);

        is_loaded = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading dictionary C++ files: " << e.what() << std::endl;
        return false;
    }
}

bool VietphraseTrie::get_meaning_at_id(uint32_t m_id, uint8_t& out_tier, std::vector<std::string>& out_meanings, std::set<std::string>& out_hv_set) {
    if (m_id >= tiers.size() || m_id + 1 >= offsets.size()) return false;
    out_tier = tiers[m_id];
    uint32_t st = offsets[m_id];
    uint32_t ed = offsets[m_id + 1];

    if (ed > flat_bytes.size() || st > ed) return false;

    std::string m_str(&flat_bytes[st], ed - st);
    out_meanings.clear();
    out_hv_set.clear();

    std::stringstream ss(m_str);
    std::string token;
    while (std::getline(ss, token, '\0')) {
        out_meanings.push_back(token);
        if (out_tier == 0) {
            out_hv_set.insert(token);
        }
    }
    return true;
}

std::vector<MatchInfo> VietphraseTrie::match_sentence(const std::string& sentence) {
    std::vector<MatchInfo> result;
    if (!is_loaded || sentence.empty()) return result;

    std::vector<std::string> chars = utf8_to_chars(sentence);
    int n = chars.size();
    if (n == 0) return result;

    std::vector<std::vector<MatchInfo>> all_matches_from(n);

    // Rank pattern matching — manual UTF-8 char-level scan
    // (std::regex cannot handle CJK character classes correctly on raw UTF-8 bytes)
    std::unordered_map<std::string, std::string> unit_map = {
        {"重", "trọng"}, {"层", "tầng"}, {"阶", "giai"}, {"品", "phẩm"}, {"级", "cấp"}, {"段", "đoạn"}, {"转", "chuyển"}
    };
    std::unordered_map<std::string, std::string> num_map = {
        {"零", "không"}, {"一", "nhất"}, {"二", "nhị"}, {"三", "tam"}, {"四", "tứ"}, {"五", "ngũ"},
        {"六", "lục"}, {"七", "thất"}, {"八", "bát"}, {"九", "cửu"}, {"十", "thập"}, {"百", "bách"}, {"千", "thiên"}, {"万", "vạn"}
    };
    std::unordered_set<std::string> num_chars_set = {"零","一","二","三","四","五","六","七","八","九","十","百","千","万"};

    auto is_ascii_digit = [](const std::string& ch) -> bool {
        return ch.length() == 1 && ch[0] >= '0' && ch[0] <= '9';
    };

    for (int i = 0; i < n; ++i) {
        // Try to find a number run (digits or CJK number chars) followed by a unit
        int num_start = i;
        int num_end = i;
        bool has_digit = false, has_cn_num = false;

        // Consume ASCII digits or CJK number characters
        while (num_end < n) {
            if (is_ascii_digit(chars[num_end])) { has_digit = true; num_end++; }
            else if (num_chars_set.count(chars[num_end])) { has_cn_num = true; num_end++; }
            else break;
        }

        if (num_end == num_start) continue; // no number found
        if (has_digit && has_cn_num) continue; // mixed — skip

        // Check for unit immediately after the number
        if (num_end >= n) continue;

        std::string unit_str;
        int unit_len = 0;
        // Check 2-char unit "重天" first
        if (num_end + 1 < n && chars[num_end] == "重" && chars[num_end + 1] == "天") {
            unit_str = "重天"; unit_len = 2;
        } else if (unit_map.count(chars[num_end])) {
            unit_str = chars[num_end]; unit_len = 1;
        }

        if (unit_len == 0) continue; // no unit found

        std::string vi_unit = (unit_str == "重天") ? "trọng thiên" : unit_map[unit_str];

        // Build vi_num
        std::string vi_num;
        std::string raw_key;
        for (int k = num_start; k < num_end; ++k) {
            if (num_map.count(chars[k])) vi_num += num_map[chars[k]];
            else vi_num += chars[k];
            raw_key += chars[k];
        }
        for (int k = 0; k < unit_len; ++k) raw_key += chars[num_end + k];

        int char_ed = num_end + unit_len;

        MatchInfo match;
        match.start = num_start;
        match.end = char_ed;
        match.length = char_ed - num_start;
        match.tier = 1;
        match.key = raw_key;
        match.meanings = {vi_num + " " + vi_unit};
        all_matches_from[num_start].push_back(match);
    }

    // Scan for Trie Matches
    for (int i = 0; i < n; ++i) {
        std::string sub = "";
        int max_j = std::min(n, i + max_key_len);
        for (int j = i; j < max_j; ++j) {
            sub += chars[j];
            int length = j - i + 1;

            std::string search_key = sub + "\xff";
            marisa::Agent agent;
            agent.set_query(search_key.c_str(), search_key.length());

            while (trie.predictive_search(agent)) {
                const marisa::Key& key = agent.key();
                if (key.length() == search_key.length() + 6) {
                    const unsigned char* ptr = (const unsigned char*)key.ptr();
                    size_t val_pos = key.length() - 6;
                    uint16_t tier = (ptr[val_pos] << 8) | ptr[val_pos + 1];
                    uint32_t m_id = (ptr[val_pos + 2] << 24) | (ptr[val_pos + 3] << 16) | (ptr[val_pos + 4] << 8) | ptr[val_pos + 5];

                    uint8_t stored_tier;
                    std::vector<std::string> meanings;
                    std::set<std::string> hv_set;
                    if (get_meaning_at_id(m_id, stored_tier, meanings, hv_set)) {
                        MatchInfo match;
                        match.start = i;
                        match.end = j + 1;
                        match.length = length;
                        match.tier = tier;
                        match.key = sub;
                        match.meanings = meanings;
                        match.hv_meanings = hv_set;
                        all_matches_from[i].push_back(match);
                    }
                }
            }
        }
    }

    // Dynamic Programming (DP) Path Reconstruction
    std::vector<double> dp(n + 1, 0.0);
    std::vector<std::pair<int, int>> parent(n + 1, {-1, -1});

    std::unordered_map<int, double> tier_multiplier = {{0, 50.0}, {1, 500.0}, {2, 200.0}, {3, 100.0}};

    for (int i = 1; i <= n; ++i) {
        dp[i] = dp[i - 1];
        parent[i] = {i - 1, -1};

        for (int j = 0; j < i; ++j) {
            for (size_t m_idx = 0; m_idx < all_matches_from[j].size(); ++m_idx) {
                const auto& m = all_matches_from[j][m_idx];
                if (m.end == i) {
                    int L = m.length;
                    double tier_bonus = tier_multiplier.count(m.tier) ? tier_multiplier[m.tier] : 0.0;
                    double w_len = 0.0;
                    if (L == 1) w_len = 1.0;
                    else if (L == 2) w_len = 20.0;
                    else if (L == 3) w_len = 120.0;
                    else if (L == 4) w_len = 400.0;
                    else w_len = L * 200.0;

                    double weight = w_len * 100.0 + L * tier_bonus + L;
                    double score = dp[j] + weight;
                    if (score > dp[i]) {
                        dp[i] = score;
                        parent[i] = {j, (int)m_idx};
                    }
                }
            }
        }
    }

    // Reconstruct DP Path
    std::vector<MatchInfo> matches;
    int curr = n;
    while (curr > 0) {
        auto p = parent[curr];
        if (p.second != -1) {
            matches.push_back(all_matches_from[p.first][p.second]);
        }
        curr = p.first;
    }

    std::reverse(matches.begin(), matches.end());
    return matches;
}
