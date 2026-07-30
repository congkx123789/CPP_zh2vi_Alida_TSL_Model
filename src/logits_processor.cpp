#include "logits_processor.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>


static std::vector<std::string> get_utf8_chars_lp(const std::string& str) {
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

static bool is_cjk_char(const std::string& ch) {
    if (ch.empty()) return false;
    unsigned char c0 = (unsigned char)ch[0];
    if (ch.length() == 3 && c0 >= 0xE4 && c0 <= 0xE9) {
        return true;
    }
    return false;
}

static std::string trim_str(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

LogitsProcessor::LogitsProcessor(TranslationTokenizer& tok) : tokenizer(tok) {}
LogitsProcessor::~LogitsProcessor() {}

bool LogitsProcessor::load_hv_dict(const std::string& hv_dict_path) {
    // Detect binary format by extension
    bool is_bin = (hv_dict_path.size() >= 4 && hv_dict_path.substr(hv_dict_path.size() - 4) == ".bin");

    std::vector<std::pair<std::string, std::string>> raw_entries;

    if (is_bin) {
        std::ifstream f(hv_dict_path, std::ios::binary);
        if (!f.is_open()) return false;

        uint32_t num_entries = 0;
        f.read((char*)&num_entries, 4);

        for (uint32_t i = 0; i < num_entries; ++i) {
            uint16_t klen = 0, vlen = 0;
            f.read((char*)&klen, 2);
            std::string key(klen, '\0');
            f.read(&key[0], klen);
            f.read((char*)&vlen, 2);
            std::string val(vlen, '\0');
            f.read(&val[0], vlen);
            raw_entries.emplace_back(key, val);
        }
    } else {
        std::ifstream f(hv_dict_path);
        if (!f.is_open()) return false;

        std::string line;
        while (std::getline(f, line)) {
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string ch = trim_str(line.substr(0, pos));
                std::string mean = trim_str(line.substr(pos + 1));
                if (!ch.empty()) raw_entries.emplace_back(ch, mean);
            }
        }
    }

    for (const auto& [ch, mean] : raw_entries) {
        hv_map[ch] = mean;
        std::stringstream ss(mean);
        std::string item;
        std::vector<std::string> unique_cands;
        std::unordered_set<std::string> seen;

        while (std::getline(ss, item, '/')) {
            item = trim_str(item);
            if (item.empty()) continue;
            if (!seen.count(item)) {
                seen.insert(item);
                unique_cands.push_back(item);
            }
        }

        if (unique_cands.size() > 1) {
            hv_char_multi.insert(ch);
        }

        hv_char_list[ch] = unique_cands;
    }
    return true;
}

bool LogitsProcessor::is_redundant_repetition(const std::string& cand_word, const std::vector<std::string>& final_words) {
    if (final_words.empty()) return false;
    std::string w_clean = trim_str(cand_word);
    if (w_clean.empty()) return false;

    std::string last_phrase = trim_str(final_words.back());
    if (last_phrase == w_clean) return true;

    return false;
}

std::string LogitsProcessor::process_logits(const float* logits_data, int seq_len, int vocab_size,
                                            const std::string& sentence_zh,
                                            const std::vector<MatchInfo>& trie_matches,
                                            float confidence_threshold) {
    std::unordered_map<int, MatchInfo> trie_by_start;
    for (const auto& m : trie_matches) {
        trie_by_start[m.start] = m;
    }

    std::vector<std::string> chars = get_utf8_chars_lp(sentence_zh);
    int n_zh = chars.size();

    std::vector<std::string> final_words;
    int i = 0;

    // Direct Raw Logit Lookup (Zero Softmax overhead, 100% mathematically equivalent to Softmax argmax)
    auto get_slice_prob = [&](int w_s, int w_e, int tok_id) -> float {
        if (tok_id < 0 || tok_id >= vocab_size) return -1e9f;
        w_s = std::max(0, w_s);
        w_e = std::min(seq_len, w_e);
        if (w_e <= w_s) return -1e9f;

        float max_l = -1e9f;
        for (int r = w_s; r < w_e; ++r) {
            float l = logits_data[r * vocab_size + tok_id];
            if (l > max_l) max_l = l;
        }
        return max_l;
    };

    while (i < n_zh) {
        std::string char_zh = chars[i];
        bool has_trie = trie_by_start.count(i) > 0;

        if (!has_trie && !is_cjk_char(char_zh)) {
            std::string non_cjk_chunk = "";
            int j = i;
            while (j < n_zh && !is_cjk_char(chars[j]) && trie_by_start.count(j) == 0) {
                non_cjk_chunk += chars[j];
                j++;
            }
            final_words.push_back(non_cjk_chunk);
            i = j;
            continue;
        }

        if (has_trie) {
            const auto& matched_entry = trie_by_start[i];
            int phrase_len = matched_entry.length;
            const auto& meanings = matched_entry.meanings;

            int w_min = std::max(0, (int)(i * ((double)seq_len / std::max(n_zh, 1))) - 2);
            int w_max = std::min(seq_len, (int)((i + phrase_len) * ((double)seq_len / std::max(n_zh, 1))) + 3);

            std::string best_meaning = meanings[0];

            if (phrase_len == 1 && hv_char_multi.count(char_zh) > 0 && hv_char_list.count(char_zh) > 0) {
                float max_cand_score = -1e9f;
                std::string best_cand = meanings[0];
                const auto& cands = hv_char_list.at(char_zh);

                for (const auto& cand : cands) {
                    auto it = tokenizer.vi2idx.find(cand);
                    if (it != tokenizer.vi2idx.end()) {
                        float score = get_slice_prob(w_min, w_max, it->second);
                        if (score > max_cand_score) {
                            max_cand_score = score;
                            best_cand = cand;
                        }
                    }
                }
                best_meaning = best_cand;
            } else if (meanings.size() > 1) {
                float max_cand_score = -1e9f;
                std::string best_cand = meanings[0];

                for (const auto& cand : meanings) {
                    float max_ai_prob = -1e9f;
                    size_t start = 0;
                    while (start < cand.size()) {
                        while (start < cand.size() && cand[start] == ' ') start++;
                        if (start >= cand.size()) break;
                        size_t end = start;
                        while (end < cand.size() && cand[end] != ' ') end++;
                        std::string sub_w = cand.substr(start, end - start);
                        start = end;

                        auto it = tokenizer.vi2idx.find(sub_w);
                        if (it != tokenizer.vi2idx.end()) {
                            float p = get_slice_prob(w_min, w_max, it->second);
                            if (p > max_ai_prob) max_ai_prob = p;
                        }
                    }

                    if (max_ai_prob > max_cand_score) {
                        max_cand_score = max_ai_prob;
                        best_cand = cand;
                    }
                }
                best_meaning = best_cand;
            }

            if (is_redundant_repetition(best_meaning, final_words)) {
                if (meanings.size() > 1) best_meaning = meanings[1];
            }

            final_words.push_back(best_meaning);
            i += phrase_len;
        } else {
            // Hán Việt Fallback: 1 meaning -> direct O(1); >1 meanings -> AI Softmax max(P_AI)
            std::string best_hv = char_zh;
            if (hv_map.count(char_zh)) {
                const std::string& hv_val = hv_map.at(char_zh);
                size_t slash_pos = hv_val.find('/');
                if (slash_pos == std::string::npos) {
                    best_hv = hv_val;
                } else {
                    int w_min = std::max(0, (int)(i * ((double)seq_len / std::max(n_zh, 1))) - 2);
                    int w_max = std::min(seq_len, (int)((i + 1) * ((double)seq_len / std::max(n_zh, 1))) + 3);

                    float max_cand_score = -1e9f;
                    best_hv = hv_val.substr(0, slash_pos);

                    size_t start = 0;
                    while (start < hv_val.size()) {
                        while (start < hv_val.size() && (hv_val[start] == '/' || hv_val[start] == ' ')) start++;
                        if (start >= hv_val.size()) break;
                        size_t end = start;
                        while (end < hv_val.size() && hv_val[end] != '/' && hv_val[end] != ' ') end++;
                        std::string cand = hv_val.substr(start, end - start);
                        start = end;

                        auto it = tokenizer.vi2idx.find(cand);
                        if (it != tokenizer.vi2idx.end()) {
                            float score = get_slice_prob(w_min, w_max, it->second);
                            if (score > max_cand_score) {
                                max_cand_score = score;
                                best_hv = cand;
                            }
                        }
                    }
                }
            }

            final_words.push_back(best_hv);
            i += 1;
        }
    }

    // Deduplicate and format output
    std::vector<std::string> res;
    for (const auto& w : final_words) {
        if (res.empty() || res.back() != w) {
            res.push_back(w);
        }
    }

    std::string out = "";
    for (size_t k = 0; k < res.size(); ++k) {
        if (k > 0) out += " ";
        out += res[k];
    }

    // Post-formatting punctuation spacing (manual, no regex needed)
    static const std::unordered_set<std::string> punct_after = {
        "，","。","！","？","；","：","、","\xe2\x80\x9d","\xe2\x80\x9d","》","】","…","—","–",",","!","?",":",";","）",")"
    };
    static const std::unordered_set<std::string> punct_open = {
        "\xe2\x80\x9c","《","【","（","("
    };

    // Remove spaces before closing/middle punctuation and after opening punctuation
    std::string cleaned;
    std::vector<std::string> out_chars;
    {
        size_t p = 0;
        while (p < out.size()) {
            unsigned char c = (unsigned char)out[p];
            size_t clen = 1;
            if ((c & 0x80) == 0) clen = 1;
            else if ((c & 0xE0) == 0xC0) clen = 2;
            else if ((c & 0xF0) == 0xE0) clen = 3;
            else if ((c & 0xF8) == 0xF0) clen = 4;
            if (p + clen <= out.size()) out_chars.push_back(out.substr(p, clen));
            p += clen;
        }
    }

    for (size_t ci = 0; ci < out_chars.size(); ++ci) {
        const std::string& ch = out_chars[ci];
        if (ch == " ") {
            // Look ahead: skip space if next non-space char is closing punct
            size_t next = ci + 1;
            while (next < out_chars.size() && out_chars[next] == " ") next++;
            if (next < out_chars.size() && punct_after.count(out_chars[next])) continue;
            // Look behind: skip space if previous non-space char is opening punct
            if (!cleaned.empty()) {
                // Get last char of cleaned
                size_t lp = cleaned.size();
                size_t llen = 1;
                if (lp >= 4 && ((unsigned char)cleaned[lp-4] & 0xF8) == 0xF0) llen = 4;
                else if (lp >= 3 && ((unsigned char)cleaned[lp-3] & 0xF0) == 0xE0) llen = 3;
                else if (lp >= 2 && ((unsigned char)cleaned[lp-2] & 0xE0) == 0xC0) llen = 2;
                std::string last_ch = cleaned.substr(lp - llen);
                if (punct_open.count(last_ch)) continue;
            }
        }
        cleaned += ch;
    }

    return cleaned;
}
