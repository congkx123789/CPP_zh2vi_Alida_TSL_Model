#ifndef DICTIONARY_HPP
#define DICTIONARY_HPP

#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <memory>
#include "marisa.h"

struct MatchInfo {
    int start;
    int end;
    int length;
    int tier;
    std::string key;
    std::vector<std::string> meanings;
    std::set<std::string> hv_meanings;
};

class VietphraseTrie {
public:
    VietphraseTrie();
    ~VietphraseTrie();

    bool load(const std::string& data_dir);
    std::vector<MatchInfo> match_sentence(const std::string& sentence);

    bool get_meaning_at_id(uint32_t m_id, uint8_t& out_tier, std::vector<std::string>& out_meanings, std::set<std::string>& out_hv_set);

private:
    marisa::Trie trie;
    std::vector<char> flat_bytes;
    std::vector<uint32_t> offsets;
    std::vector<uint8_t> tiers;
    int max_key_len;
    bool is_loaded;
};

#endif // DICTIONARY_HPP
