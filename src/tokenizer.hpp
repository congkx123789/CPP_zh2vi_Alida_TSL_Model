#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

class TranslationTokenizer {
public:
    TranslationTokenizer();
    ~TranslationTokenizer();

    bool load(const std::string& data_dir);
    std::vector<int64_t> encode_zh(const std::string& text, int max_len = 64);
    std::string decode_vi(const std::vector<int64_t>& token_ids);

    std::unordered_map<std::string, int32_t> zh2idx;
    std::unordered_map<std::string, int32_t> vi2idx;
    std::unordered_map<int32_t, std::string> idx2vi;

    int32_t pad_id;
    int32_t unk_id;
    int32_t bos_id;
    int32_t eos_id;
    bool is_loaded;
};

#endif // TOKENIZER_HPP
