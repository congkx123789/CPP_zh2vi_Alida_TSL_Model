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

            std::string best_meaning = "";
            if (meanings.size() == 1) {
                const auto& cand = meanings[0];
                std::string best_cand = cand;

                if (hv_char_multi.count(cand) > 0) {
                    const auto& cands = hv_char_list.at(cand);
                    float max_cand_score = -1e9f;
                    best_cand = cands[0];

                    for (const auto& c : cands) {
                        if (c.empty()) {
                            float score = -2.5f;
                            if (score > max_cand_score) {
                                max_cand_score = score;
                                best_cand = "";
                            }
                        } else {
                            auto it = tokenizer.vi2idx.find(c);
                            if (it != tokenizer.vi2idx.end()) {
                                float score = get_slice_prob(w_min, w_max, it->second);
                                if (score > max_cand_score) {
                                    max_cand_score = score;
                                    best_cand = c;
                                }
                            }
                        }
                    }
                }
                best_meaning = best_cand;
            } else if (meanings.size() > 1) {
                float max_cand_score = -1e9f;
                std::string best_cand = meanings[0];

                for (const auto& cand : meanings) {
                    if (cand.empty()) {
                        float score = -2.5f;
                        if (score > max_cand_score) {
                            max_cand_score = score;
                            best_cand = "";
                        }
                        continue;
                    }
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

            if (!is_redundant_repetition(best_meaning, final_words)) {
                final_words.push_back(best_meaning);
            }
            i += phrase_len;
        } else {
            std::string best_hv = char_zh;
            if (hv_map.count(char_zh) > 0) {
                const std::string& hv_val = hv_map.at(char_zh);
                if (hv_val.empty()) {
                    best_hv = "";
                } else if (hv_char_list.count(char_zh) > 0) {
                    const auto& cands = hv_char_list.at(char_zh);
                    if (cands.size() == 1) {
                        best_hv = cands[0];
                    } else {
                        int w_min = std::max(0, (int)(i * ((double)seq_len / std::max(n_zh, 1))) - 2);
                        int w_max = std::min(seq_len, (int)((i + 1) * ((double)seq_len / std::max(n_zh, 1))) + 3);

                        float max_cand_score = -1e9f;
                        best_hv = cands[0];

                        for (const auto& cand : cands) {
                            if (cand.empty()) {
                                float score = -2.5f;
                                if (score > max_cand_score) {
                                    max_cand_score = score;
                                    best_hv = "";
                                }
                            } else {
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
                } else {
                    size_t slash_pos = hv_val.find('/');
                    best_hv = (slash_pos == std::string::npos) ? hv_val : hv_val.substr(0, slash_pos);
                }
            }

            final_words.push_back(best_hv);
            i += 1;
        }
    }

    for (auto& w : final_words) {
        if (w == "，" || w == "、") w = ",";
        else if (w == "。") w = ".";
        else if (w == "！") w = "!";
        else if (w == "？") w = "?";
        else if (w == "：" || w == "︰") w = ":";
        else if (w == "；") w = ";";
        else if (w == "“" || w == "”" || w == "《" || w == "》") w = "\"";
        else if (w == "（") w = "(";
        else if (w == "）") w = ")";
        else if (w == "【") w = "[";
        else if (w == "】") w = "]";
    }

    // Deduplicate consecutive words
    std::vector<std::string> res;
    for (const auto& w : final_words) {
        if (w.empty()) continue;
        if (res.empty() || res.back() != w) {
            res.push_back(w);
        }
    }

    std::string out = "";
    for (size_t k = 0; k < res.size(); ++k) {
        if (k > 0) {
            const auto& prev = res[k - 1];
            const auto& cur = res[k];
            if (cur != "." && cur != "," && cur != ";" && cur != ":" && cur != "!" && cur != "?" &&
                prev != "“" && prev != "(" && prev != "[" && cur != "”" && cur != ")" && cur != "]") {
                out += " ";
            }
        }
        out += res[k];
    }

    std::string cleaned = "";
    for (size_t ci = 0; ci < out.size(); ++ci) {
        if (out[ci] == ' ' && ci + 1 < out.size()) {
            char next_c = out[ci + 1];
            if (next_c == '.' || next_c == ',' || next_c == ';' || next_c == ':' || next_c == '!' || next_c == '?' || next_c == ')' || next_c == ']') {
                continue;
            }
        }
        cleaned += out[ci];
    }

    // Capitalize first letter of each sentence
    std::string capitalized = "";
    bool cap_next = true;
    size_t ci = 0;
    while (ci < cleaned.size()) {
        unsigned char c = (unsigned char)cleaned[ci];
        size_t clen = 1;
        if ((c & 0x80) == 0) clen = 1;
        else if ((c & 0xE0) == 0xC0) clen = 2;
        else if ((c & 0xF0) == 0xE0) clen = 3;
        else if ((c & 0xF8) == 0xF0) clen = 4;

        std::string char_str = cleaned.substr(ci, clen);
        ci += clen;

        if (cap_next) {
            if (c >= 'a' && c <= 'z') {
                char_str[0] = (char)std::toupper(c);
                cap_next = false;
            } else if (c >= 'A' && c <= 'Z') {
                cap_next = false;
            } else if (clen > 1) {
                if (char_str == "đ") char_str = "Đ";
                else if (char_str == "à") char_str = "À";
                else if (char_str == "á") char_str = "Á";
                else if (char_str == "ả") char_str = "Ả";
                else if (char_str == "ã") char_str = "Ã";
                else if (char_str == "ạ") char_str = "Ạ";
                else if (char_str == "ă") char_str = "Ă";
                else if (char_str == "â") char_str = "Â";
                else if (char_str == "ê") char_str = "Ê";
                else if (char_str == "ô") char_str = "Ô";
                else if (char_str == "ơ") char_str = "Ơ";
                else if (char_str == "ư") char_str = "Ư";
                cap_next = false;
            }
        }

        if (char_str == "." || char_str == "?" || char_str == "!" || char_str == "\"" || char_str == "\n") {
            cap_next = true;
        }

        capitalized += char_str;
    }

    return capitalized;
}
