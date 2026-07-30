#include "tokenizer.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

static std::vector<std::string> get_utf8_chars(const std::string& str) {
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

TranslationTokenizer::TranslationTokenizer() : is_loaded(false), pad_id(0), unk_id(1), bos_id(2), eos_id(3) {}
TranslationTokenizer::~TranslationTokenizer() {}

bool TranslationTokenizer::load(const std::string& data_dir) {
    std::string zh_vocab_path = data_dir + "/zh_vocab.bin";
    std::string vi_vocab_path = data_dir + "/vi_vocab.bin";

    {
        std::ifstream f(zh_vocab_path, std::ios::binary);
        if (!f.is_open()) return false;
        uint32_t count = 0;
        f.read((char*)&count, 4);

        for (uint32_t i = 0; i < count; ++i) {
            uint16_t len = 0;
            f.read((char*)&len, 2);
            std::string key(len, '\0');
            f.read(&key[0], len);
            uint32_t idx = 0;
            f.read((char*)&idx, 4);
            zh2idx[key] = idx;
        }
    }

    {
        std::ifstream f(vi_vocab_path, std::ios::binary);
        if (!f.is_open()) return false;
        uint32_t count = 0;
        f.read((char*)&count, 4);

        for (uint32_t i = 0; i < count; ++i) {
            uint16_t len = 0;
            f.read((char*)&len, 2);
            std::string key(len, '\0');
            f.read(&key[0], len);
            uint32_t idx = 0;
            f.read((char*)&idx, 4);
            vi2idx[key] = idx;
            idx2vi[idx] = key;
        }
    }

    pad_id = zh2idx.count("[PAD]") ? zh2idx["[PAD]"] : 0;
    unk_id = zh2idx.count("[UNK]") ? zh2idx["[UNK]"] : 1;
    bos_id = zh2idx.count("[BOS]") ? zh2idx["[BOS]"] : 2;
    eos_id = zh2idx.count("[EOS]") ? zh2idx["[EOS]"] : 3;

    is_loaded = true;
    return true;
}

std::vector<int64_t> TranslationTokenizer::encode_zh(const std::string& text, int max_len) {
    std::vector<std::string> chars = get_utf8_chars(text);
    std::vector<int64_t> tokens;

    int n = std::min((int)chars.size(), max_len);
    for (int i = 0; i < n; ++i) {
        if (zh2idx.count(chars[i])) {
            tokens.push_back(zh2idx[chars[i]]);
        } else {
            tokens.push_back(unk_id);
        }
    }

    // Pad to exact max_len for ONNX Tensor input shape alignment
    while ((int)tokens.size() < max_len) {
        tokens.push_back(pad_id);
    }

    return tokens;
}

std::string TranslationTokenizer::decode_vi(const std::vector<int64_t>& token_ids) {
    std::string result = "";
    for (auto tid : token_ids) {
        if (tid == pad_id || tid == bos_id || tid == eos_id) continue;
        if (idx2vi.count((int32_t)tid)) {
            if (!result.empty()) result += " ";
            result += idx2vi[(int32_t)tid];
        }
    }
    return result;
}
