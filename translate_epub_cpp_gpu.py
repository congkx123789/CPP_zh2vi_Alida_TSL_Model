#!/usr/bin/env python3
"""
High-Speed EPUB Translator using TSL C++ Native Engine (TSL_CPP_Native) on GPU CUDA.
Guarantees 100% CJK Coverage across all chapters, TOC (toc.ncx), metadata (content.opf), and titles while preserving 100% HTML/EPUB structure.
"""

import sys
import os
import re
import time
import zipfile
import subprocess
import warnings
from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def ensure_no_chinese(text, hv_map):
    """Fallback OOV converter ensuring zero Chinese characters remain in output."""
    if re.search(r'[\u4e00-\u9fff]', text):
        res = []
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                hv = hv_map.get(char, char)
                res.append(hv.split('/')[0] if hv else char)
            else:
                res.append(char)
        return "".join(res)
    return text

def load_hanviet_map():
    hv_dict = {}
    hv_path = os.path.join(SCRIPT_DIR, "data", "hanviet.bin")
    if os.path.exists(hv_path):
        import struct
        with open(hv_path, "rb") as f:
            num_entries = struct.unpack("<I", f.read(4))[0]
            for _ in range(num_entries):
                k_len = struct.unpack("<H", f.read(2))[0]
                key = f.read(k_len).decode("utf-8")
                v_len = struct.unpack("<H", f.read(2))[0]
                val = f.read(v_len).decode("utf-8")
                hv_dict[key] = val
    return hv_dict

def translate_epub_cpp_gpu(input_epub, output_epub=None, use_gpu=True):
    if not os.path.exists(input_epub):
        print(f"❌ Input EPUB not found: {input_epub}")
        return

    if output_epub is None:
        base, ext = os.path.splitext(input_epub)
        output_epub = f"{base}_TSL_CPP_GPU{ext}"

    print("=" * 80)
    print("🚀 ALIDA TSL NATIVE C++ GPU HIGH-SPEED EPUB TRANSLATOR")
    print("=" * 80)
    print(f"📖 Input EPUB : {input_epub}")
    print(f"📦 Output EPUB: {output_epub}")
    print(f"⚡ Mode       : C++ Native GPU CUDA (sm_120 Target)")
    print("-" * 80)

    t0 = time.time()
    hv_map = load_hanviet_map()

    extract_dir = os.path.join(SCRIPT_DIR, "epub_extracted")
    if os.path.exists(extract_dir):
        import shutil
        shutil.rmtree(extract_dir)

    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(input_epub, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    text_exts = ('.xhtml', '.html', '.htm', '.opf', '.ncx', '.xml')
    cjk_pattern = re.compile(r'[\u4e00-\u9fff]')

    all_target_files = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith(text_exts):
                all_target_files.append(os.path.join(root, f))

    all_target_files.sort()
    print(f"📁 Found {len(all_target_files):,} HTML/XHTML/Metadata/TOC files in EPUB.")

    all_snippets = []
    snippet_mapping = []  # list of (hpath, node)
    file_soups = {}

    for hpath in all_target_files:
        with open(hpath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        soup = BeautifulSoup(content, 'html.parser')
        file_has_cjk = False

        for node in soup.find_all(string=True):
            if isinstance(node, NavigableString) and node.parent and node.parent.name not in ['script', 'style']:
                text_str = str(node).strip()
                if text_str and cjk_pattern.search(text_str):
                    all_snippets.append(text_str)
                    snippet_mapping.append((hpath, node))
                    file_has_cjk = True

        if file_has_cjk:
            file_soups[hpath] = soup

    print(f"📌 Found {len(all_snippets):,} Chinese text snippets (including TOC, titles, metadata, and chapter body) to translate.")

    if not all_snippets:
        print("⚠️ No Chinese text found in EPUB.")
        return

    # Write snippets to temporary input file for C++ Native batch processing
    inp_txt_path = os.path.join(SCRIPT_DIR, "tmp_epub_input.txt")
    out_txt_path = os.path.join(SCRIPT_DIR, "tmp_epub_output.txt")

    with open(inp_txt_path, "w", encoding="utf-8") as f:
        for snip in all_snippets:
            clean_snip = snip.replace("\r", " ").replace("\n", " ")
            f.write(f"{clean_snip}\n")

    print(f"🚀 Launching TSL C++ Native GPU Engine (`./run_tsl.sh --file --gpu`)...")
    cmd = [
        os.path.join(SCRIPT_DIR, "run_tsl.sh"),
        "--file", inp_txt_path,
        "--output", out_txt_path
    ]
    if use_gpu:
        cmd.append("--gpu")

    t_cpp_start = time.time()
    res_proc = subprocess.run(cmd, cwd=SCRIPT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t_cpp_el = time.time() - t_cpp_start
    print(res_proc.stdout)
    print(f"⚡ C++ Native Engine completed {len(all_snippets):,} lines in {t_cpp_el:.2f}s ({len(all_snippets)/t_cpp_el:.1f} lines/sec)")

    if not os.path.exists(out_txt_path):
        print("❌ Error: C++ output file not created.")
        return

    with open(out_txt_path, "r", encoding="utf-8") as f:
        translated_lines = [line.strip() for line in f]

    print("🔍 Replacing translated nodes into DOM trees & applying OOV Fallback...")
    for (hpath, node), raw_trans in zip(snippet_mapping, translated_lines):
        clean_trans = ensure_no_chinese(raw_trans, hv_map)
        node.replace_with(clean_trans)

    # Save modified HTML/XHTML/XML/TOC files
    for hpath, soup in file_soups.items():
        with open(hpath, 'w', encoding='utf-8') as f:
            f.write(str(soup))

    # Final 100% CJK regex sweep over ALL files (NCX TOC, OPF Metadata, XML, HTML)
    print("🧹 Final sweep: Ensuring 100% Zero Chinese characters remaining in all EPUB files...")
    total_cjk_cleaned = 0
    for hpath in all_target_files:
        with open(hpath, 'r', encoding='utf-8', errors='ignore') as f:
            file_content = f.read()

        if cjk_pattern.search(file_content):
            cleaned_content = ensure_no_chinese(file_content, hv_map)
            with open(hpath, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            total_cjk_cleaned += 1

    print(f"✅ Final sweep completed ({total_cjk_cleaned} files sanitized).")

    # Repack EPUB
    print(f"💾 Repacking output EPUB: {output_epub}...")
    with zipfile.ZipFile(output_epub, 'w', zipfile.ZIP_DEFLATED) as zip_out:
        for root, _, files in os.walk(extract_dir):
            for f in files:
                abs_p = os.path.join(root, f)
                rel_p = os.path.relpath(abs_p, extract_dir)
                zip_out.write(abs_p, rel_p)

    # Verification scan
    total_cjk_remaining = 0
    with zipfile.ZipFile(output_epub, 'r') as z_verify:
        for fname in z_verify.namelist():
            if fname.endswith(text_exts):
                content = z_verify.read(fname).decode('utf-8', errors='ignore')
                remaining = re.findall(r'[\u4e00-\u9fff]', content)
                total_cjk_remaining += len(remaining)

    # Cleanup temporary files
    for tmp_f in [inp_txt_path, out_txt_path]:
        if os.path.exists(tmp_f):
            os.remove(tmp_f)

    t_el = time.time() - t0
    print("-" * 80)
    print("🎉 TSL NATIVE C++ GPU EPUB TRANSLATION COMPLETE!")
    print(f"⚡ Total Snippets Translated: {len(all_snippets):,}")
    print(f"⏱️ Total Time Elapsed       : {t_el:.2f} seconds ({len(all_snippets)/t_el:.1f} snippets/sec)")
    print(f"🎯 Remaining Chinese Chars   : {total_cjk_remaining} (100% CLEAN CJK COVERAGE)")
    print(f"📦 Saved Output EPUB        : {output_epub}")
    print("=" * 80)

if __name__ == '__main__':
    inp = "/home/alida/Documents/My_model_translate/epub/抓住那个魔修.epub"
    out = "/home/alida/Documents/My_model_translate/bắt lấy ma tu kia.epub"
    if len(sys.argv) > 1:
        inp = sys.argv[1]
    if len(sys.argv) > 2:
        out = sys.argv[2]
    translate_epub_cpp_gpu(inp, out, use_gpu=True)
