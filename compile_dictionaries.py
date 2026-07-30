#!/usr/bin/env python3
"""
Master Dictionary Compiler for Alida TSL Translation Engine.
Automatically reads all 5 text dictionary files from 'Data/my dataset',
compiles them into high-performance zero-copy binary format (MARISA Trie + Flat Buffer),
and updates both TSL_CPP_Native/data/ and TSL_Translator_Standalone/data/.
"""

import os
import sys
import time
import struct
import array
import pickle

try:
    import marisa_trie
    HAS_MARISA = True
except ImportError:
    HAS_MARISA = False
    print("❌ Error: marisa_trie package not installed. Run: pip install marisa_trie")
    sys.exit(1)

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(root_dir, "Data", "my dataset")) and os.path.exists("/home/alida/Documents/My_model_translate/Data/my dataset"):
        root_dir = "/home/alida/Documents/My_model_translate"

    dataset_dir = os.path.join(root_dir, "Data", "my dataset")
    cpp_data_dir = os.path.join(root_dir, "TSL_CPP_Native", "data")
    standalone_data_dir = os.path.join(root_dir, "TSL_Translator_Standalone", "data")

    print("=" * 80)
    print("📦 ALIDA TSL - MASTER DICTIONARY BINARY COMPILER")
    print("=" * 80)
    print(f"📂 Nguồn từ điển text gốc: {dataset_dir}")
    print(f"🎯 Đích C++ Native       : {cpp_data_dir}")
    print(f"🎯 Đích Standalone Python: {standalone_data_dir}")
    print("-" * 80)

    t0 = time.time()

    # Target directories to sync
    target_dirs = [cpp_data_dir, standalone_data_dir]
    for d in target_dirs:
        os.makedirs(d, exist_ok=True)

    # -------------------------------------------------------------------------
    # PART 1: Compile Hán-Việt Character Dictionary -> hanviet.bin
    # -------------------------------------------------------------------------
    hv_txt_path = os.path.join(dataset_dir, "HanViet_CharDict_Enhanced_Merged.txt")
    hv_entries = []
    if os.path.exists(hv_txt_path):
        with open(hv_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                if "=" in line_str:
                    key, val = line_str.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if key:
                        hv_entries.append((key, val))
        
        for d in target_dirs:
            out_hv = os.path.join(d, "hanviet.bin")
            with open(out_hv, "wb") as f:
                f.write(struct.pack("<I", len(hv_entries)))
                for key, val in hv_entries:
                    kb = key.encode("utf-8")
                    vb = val.encode("utf-8")
                    f.write(struct.pack("<H", len(kb)))
                    f.write(kb)
                    f.write(struct.pack("<H", len(vb)))
                    f.write(vb)
        print(f"✅ [Hán Việt Single Char]: Biên dịch {len(hv_entries):,} từ ➔ hanviet.bin")
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy {hv_txt_path}")

    # -------------------------------------------------------------------------
    # PART 2: Compile Multi-character Vietphrase & Aligned files -> trie.marisa, meanings_flat.bin, offsets.bin, tiers.bin
    # -------------------------------------------------------------------------
    dict_files = [
        ("Aligned_HanViet.txt", 0),
        ("Vietphrase_2_to_5.txt", 1),
        ("Vietphrase_gt_5.txt", 2),
        ("Vietphrase_Contextual.txt", 3)
    ]

    puncs = set("，。！？；：、“”《》【】…—–,!?:;")
    meaning2id = {}
    meanings_store = []
    keys = []
    values = []
    max_key_len = 1
    total_raw_bytes = 0

    for fname, tier_idx in dict_files:
        fpath = os.path.join(dataset_dir, fname)
        if not os.path.exists(fpath):
            print(f"⚠️ Cảnh báo: Không tìm thấy {fname}")
            continue

        file_sz = os.path.getsize(fpath)
        total_raw_bytes += file_sz
        count = 0

        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                if "=" in line_str:
                    key, val = line_str.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if key and not any(c in puncs for c in key):
                        key_intern = sys.intern(key)
                        max_key_len = max(max_key_len, len(key_intern))

                        if val not in meaning2id:
                            m_id = len(meanings_store)
                            meaning2id[val] = m_id
                            meanings = tuple(sys.intern(m.strip()) for m in val.split("/"))
                            hv_set = set(sys.intern(m.strip()) for m in val.split("/")) if tier_idx == 0 else set()
                            meanings_store.append((tier_idx, meanings, hv_set))
                        else:
                            m_id = meaning2id[val]

                        keys.append(key_intern)
                        values.append((tier_idx, m_id))
                        count += 1

        print(f"📥 [Tier {tier_idx}] {fname:28s}: {count:,} keys ({file_sz / (1024*1024):.2f} MB)")

    print(f"🚀 Xây dựng MARISA DAWG RecordTrie cho {len(keys):,} tổng số từ...")
    c_trie = marisa_trie.RecordTrie(">HI", zip(keys, values))

    all_bytes = []
    offsets = [0]
    tiers = bytearray()

    for tier, m_list, _ in meanings_store:
        tiers.append(tier)
        encoded = "\0".join(m_list).encode("utf-8")
        all_bytes.append(encoded)
        offsets.append(offsets[-1] + len(encoded))

    flat_bytes = b"".join(all_bytes)
    offset_array = array.array("I", offsets)

    meta = {
        "max_key_len": max_key_len,
        "total_keys": len(keys),
        "total_meanings": len(meanings_store)
    }

    for d in target_dirs:
        c_trie.save(os.path.join(d, "trie.marisa"))
        with open(os.path.join(d, "meanings_flat.bin"), "wb") as f:
            f.write(flat_bytes)
        with open(os.path.join(d, "offsets.bin"), "wb") as f:
            offset_array.tofile(f)
        with open(os.path.join(d, "tiers.bin"), "wb") as f:
            f.write(tiers)
        with open(os.path.join(d, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

    t_el = time.time() - t0
    total_bin_sz = sum(
        os.path.getsize(os.path.join(cpp_data_dir, f))
        for f in ["trie.marisa", "meanings_flat.bin", "offsets.bin", "tiers.bin", "hanviet.bin"]
        if os.path.exists(os.path.join(cpp_data_dir, f))
    ) / (1024 * 1024)

    print("-" * 80)
    print("🎉 HOÀN THÀNH BIÊN DỊCH BỘ TỪ ĐIỂN TỐI ƯU CỰC HẠN!")
    print(f"📁 Dung lượng file text gốc (.txt)           : {total_raw_bytes / (1024*1024):.2f} MB")
    print(f"📦 Dung lượng tệp nhị phân (.bin / marisa)  : {total_bin_sz:.2f} MB")
    print(f"⚡ Thời gian xử lý                           : {t_el:.2f} giây")
    print("=" * 80)

if __name__ == "__main__":
    main()
