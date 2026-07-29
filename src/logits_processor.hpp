#ifndef LOGITS_PROCESSOR_HPP
#define LOGITS_PROCESSOR_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "tokenizer.hpp"
#include "dictionary.hpp"

class LogitsProcessor {
public:
    LogitsProcessor(TranslationTokenizer& tokenizer);
    ~LogitsProcessor();

    bool load_hv_dict(const std::string& hv_dict_path);
    std::string process_logits(const float* logits_data, int seq_len, int vocab_size,
                               const std::string& sentence_zh,
                               const std::vector<MatchInfo>& trie_matches,
                               float confidence_threshold = 0.40f);

private:
    TranslationTokenizer& tokenizer;
    std::unordered_map<std::string, std::string> hv_map;
    std::unordered_map<std::string, std::vector<std::string>> hv_char_list;
    std::unordered_map<std::string, std::unordered_set<std::string>> hv_pure_set;
    std::unordered_map<std::string, std::unordered_set<std::string>> hv_tv_set;
    std::unordered_set<std::string> hv_char_multi;

    bool is_redundant_repetition(const std::string& cand_word, const std::vector<std::string>& final_words);
};

#endif // LOGITS_PROCESSOR_HPP
