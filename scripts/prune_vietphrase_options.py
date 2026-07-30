#!/usr/bin/env python3
"""
Script to prune Vietphrase entries that have > 5 translation options down to the top 5
based on length matching with the Chinese key, word count quality, and position priority.
Entries with <= 5 options remain untouched.
"""

import sys
import shutil

def score_option(zh_key, option, original_index):
    zh_len = len(zh_key)
    words = option.strip().split()
    word_count = len(words)
    
    if word_count == 0:
        return -999.0

    # 1. Word count difference relative to Chinese key length
    diff = abs(word_count - zh_len)
    len_score = 10.0 - (diff * 2.5)
    
    # 2. Penalty for single-word vague noise when Chinese key is 2+ chars
    noise_penalty = 0.0
    if zh_len >= 2 and word_count == 1:
        noise_penalty += 3.0
        
    # 3. Penalty for bloated paraphrases (word_count > zh_len + 2)
    if word_count > zh_len + 2:
        noise_penalty += 3.5
        
    # 4. Penalty for ellipsis in translation
    if "..." in option:
        noise_penalty += 0.5
        
    # 5. Position score (earlier in original file = higher priority)
    pos_score = max(0.0, 3.0 - original_index * 0.15)
    
    return len_score - noise_penalty + pos_score

def prune_line(line):
    line_str = line.strip()
    if not line_str or line_str.startswith("#") or "=" not in line_str:
        return line if line.endswith("\n") else line + "\n"
        
    key, val = line_str.split("=", 1)
    raw_opts = [o.strip() for o in val.split("/") if o.strip()]
    
    # Remove exact duplicates within the line
    seen = set()
    opts = []
    for o in raw_opts:
        if o not in seen:
            seen.add(o)
            opts.append(o)
            
    if len(opts) <= 5:
        return f"{key}=" + "/".join(opts) + "\n"
        
    # Score & rank if > 5 options
    scored = []
    for idx, opt in enumerate(opts):
        sc = score_option(key, opt, idx)
        scored.append((opt, sc, idx))
        
    scored.sort(key=lambda x: (-x[1], x[2]))
    
    top_5 = [item[0] for item in scored[:5]]
    return f"{key}=" + "/".join(top_5) + "\n"

def process_file(file_path):
    backup_path = file_path + ".bak_prune_5"
    print(f"Creating backup at {backup_path}...")
    shutil.copyfile(file_path, backup_path)
    
    output_lines = []
    pruned_count = 0
    total_entries = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.strip().startswith("#") and "=" in line:
                total_entries += 1
                key, val = line.strip().split("=", 1)
                opts = [o.strip() for o in val.split("/") if o.strip()]
                if len(opts) > 5:
                    pruned_count += 1
            new_line = prune_line(line)
            output_lines.append(new_line)
            
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)
        
    print(f"Processed {total_entries} entries.")
    print(f"Pruned {pruned_count} entries with > 5 options down to top 5.")
    print("Done!")

if __name__ == "__main__":
    target = "/home/alida/Documents/My_model_translate/Data/my dataset/Vietphrase_2_to_5.txt"
    if len(sys.argv) > 1:
        target = sys.argv[1]
    process_file(target)
