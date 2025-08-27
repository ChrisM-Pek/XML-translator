#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys, json, time, unicodedata
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from datetime import datetime
import requests
from lxml import etree

# ===== Config =====
INPUT_DIR   = Path(os.getenv("INPUT_DIR", "input"))
OUTPUT_DIR  = Path(os.getenv("OUTPUT_DIR", "output"))
TARGET_LANG = os.getenv("TARGET_LANG", "FR")
ENGINE      = os.getenv("ENGINE", "auto")  # auto|lmstudio|deepl

# LM Studio
LMSTUDIO_API_BASE = os.getenv("LMSTUDIO_API_BASE", "http://localhost:1234/v1")
LMSTUDIO_API_KEY  = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
LMSTUDIO_MODEL    = os.getenv("LMSTUDIO_MODEL", "")

# DeepL
DEEPL_API_URLS = {"free": "https://api-free.deepl.com/v2/translate",
                  "pro":  "https://api.deepl.com/v2/translate"}
DEEPL_API_TIER = os.getenv("DEEPL_API_TIER", "free")

# ===== Logs =====
def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def fmt_dur(seconds: float) -> str:
    m, s = divmod(int(max(0, seconds)), 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h{m:02d}m{s:02d}s"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"

# ===== Placeholders =====
PLACEHOLDER_PATTERNS = [
    r"\{[^\}]*\}",   # {name}, {0}
    r"%\w",          # %s, %d
    r"\$?\d+",       # 1, $1
    r"\[[^\]]+\]",   # [CTRL], [E]
    r"<[^>]+>",      # balises HTML/XML inline
]
PLACEHOLDER_REGEX = re.compile("|".join(f"({p})" for p in PLACEHOLDER_PATTERNS))

def protect_placeholders(text: str):
    mapping: Dict[str, str] = {}
    def _repl(m):
        token = f"§§PH{len(mapping)}§§"
        mapping[token] = m.group(0)
        return token
    return PLACEHOLDER_REGEX.sub(_repl, text), mapping

def unprotect_placeholders(text: str, mapping: Dict[str, str]) -> str:
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text

# ===== Sanitizer =====
_THINK_BLOCKS = [
    (re.compile(r"(?is)<\s*think\s*>.*?<\s*/\s*think\s*>") , ""),
    (re.compile(r"(?is)&lt;\s*think\s*&gt;.*?&lt;\s*/\s*think\s*&gt;"), ""),
]
_META_LINES = re.compile(r"(?mi)^\s*(?:thought|reasoning|analysis|system|assistant|user)\s*:\s*.*$")

def sanitize_model_output(out: str) -> str:
    if not out: return out
    for rx, repl in _THINK_BLOCKS: out = rx.sub(repl, out)
    out = _META_LINES.sub("", out)
    out = out.strip().strip("`").strip()
    if (out.startswith('"') and out.endswith('"')) or (out.startswith("'") and out.endswith("'")):
        out = out[1:-1].strip()
    return out

def too_verbose(inp: str, out: str, ratio: float = 3.0) -> bool:
    return len(out) > ratio * max(1, len(inp))

# ===== XML Parser =====
def make_xml_parser():
    return etree.XMLParser(remove_blank_text=False, recover=True, huge_tree=True, resolve_entities=False)

# ===== Postprocess langue =====
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c)!="Mn")

def postprocess_output_for_lang(text: str, target_lang: str) -> str:
    if (target_lang or "").upper().startswith("FR"):
        return strip_accents(text)   # garde le comportement existant (sans accents)
    return text

# ===== Moteurs =====
def choose_engine() -> str:
    if ENGINE.lower() in ("lmstudio","deepl"):
        return ENGINE.lower()
    try:
        url = f"{LMSTUDIO_API_BASE}/models"
        headers = {"Content-Type": "application/json"}
        if LMSTUDIO_API_KEY:
            headers["Authorization"] = f"Bearer {LMSTUDIO_API_KEY}"
        r = requests.get(url, headers=headers, timeout=1.5)
        if r.ok:
            return "lmstudio"
    except Exception:
        pass
    if os.getenv("DEEPL_API_KEY"):
        return "deepl"
    raise RuntimeError("Aucun moteur dispo. Lance LM Studio OU exporte DEEPL_API_KEY.")

def _lmstudio_chat(prompt_user: str, target_lang: str) -> str:
    system_msg = (
        "You are a translation engine. "
        f"Translate the user's text into {target_lang}. "
        "Never add explanations, notes, reflections, or tags. "
        "Return ONLY the translated text."
    )
    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_user},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1024,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if LMSTUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {LMSTUDIO_API_KEY}"
    url = f"{LMSTUDIO_API_BASE}/chat/completions"
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    if not r.ok:
        raise RuntimeError(f"LM Studio error {r.status_code}: {r.text[:200]}")
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Réponse LM Studio invalide: {json.dumps(data)[:200]}") from e

def translate_lmstudio(text: str, target_lang: str = "FR") -> str:
    protected, mapping = protect_placeholders(text)
    raw = _lmstudio_chat(protected, target_lang)
    out = sanitize_model_output(raw)
    if too_verbose(protected, out, ratio=3.0):
        raise RuntimeError("LM Studio a renvoyé du contenu verbeux (protégé).")
    out = unprotect_placeholders(out, mapping)
    out = re.sub(r"(?is)</?\s*think\s*>", "", out)
    return postprocess_output_for_lang(out, target_lang)

def translate_deepl(text: str, target_lang: str = "FR") -> str:
    api_key = os.getenv("DEEPL_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPL_API_KEY manquant.")
    url = DEEPL_API_URLS.get(DEEPL_API_TIER, DEEPL_API_URLS["free"])
    protected, mapping = protect_placeholders(text)
    payload = {"text": protected, "target_lang": target_lang.upper(), "preserve_formatting": "1"}
    headers = {"Authorization": f"DeepL-Auth-Key {api_key}"}
    r = requests.post(url, data=payload, headers=headers, timeout=30)
    if not r.ok:
        raise RuntimeError(f"DeepL error {r.status_code}: {r.text[:200]}")
    data = r.json()
    try:
        out = data["translations"][0]["text"]
    except Exception as e:
        raise RuntimeError(f"Réponse DeepL inattendue: {json.dumps(data)[:200]}") from e
    out = unprotect_placeholders(out, mapping)
    return postprocess_output_for_lang(out, target_lang)

def translate_text(text: str, target_lang: str = "FR") -> str:
    if not text or text.strip() == "":
        return text
    eng = choose_engine()
    if eng == "lmstudio":
        try:
            return translate_lmstudio(text, target_lang)
        except Exception as e:
            if os.getenv("DEEPL_API_KEY"):
                log(f"    [WARN] LM Studio rejeté ({e}); fallback DeepL.")
                return translate_deepl(text, target_lang)
            raise
    else:
        return translate_deepl(text, target_lang)

# ===== Filtres =====
def is_translatable_text(s: str) -> bool:
    if not s:
        return False
    # identifiant snake_case => ignorer
    if re.fullmatch(r"[A-Za-z0-9_]+", s) and "_" in s:
        return False
    return True

# ===== Parcours XML =====
def iter_xml_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.xml"):
        if p.is_file():
            yield p

def count_text_chars_in_file(path: Path) -> Tuple[int, int]:
    try:
        parser = make_xml_parser()
        tree = etree.parse(str(path), parser)
    except Exception as e:
        log(f"[PRESCAN:PARSE_ERR] {path}: {e}")
        return 0, 0
    root = tree.getroot()
    if root is None:
        log(f"[PRESCAN:NO_ROOT] {path} : pas de racine XML valide")
        return 0, 0
    total = 0
    count = 0
    for el in root.iter("text"):
        s = (el.text or "").strip()
        if not s:
            continue
        if not is_translatable_text(s):
            continue
        count += 1
        total += len(s)
    if count == 0:
        log(f"[PRESCAN:ZERO_TRANSLATABLE] {path} : aucun <text> traduisible")
    return count, total

def prescan_dir(in_dir: Path, out_dir: Optional[Path] = None) -> Tuple[int, int, Dict[Path, Tuple[int, int]]]:
    file_stats: Dict[Path, Tuple[int, int]] = {}
    total_chars = 0
    files = []
    for f in iter_xml_files(in_dir):
        if out_dir is not None:
            rel = f.relative_to(in_dir)
            dst = out_dir / rel
            if dst.exists():
                continue
        files.append(f)
    for f in files:
        c, ch = count_text_chars_in_file(f)
        if c > 0 and ch > 0:
            file_stats[f] = (c, ch)
            total_chars += ch
    return len(file_stats), total_chars, file_stats

# ===== Purge input basée sur output =====
def purge_input_if_output_exists(in_dir: Path, out_dir: Path) -> int:
    if not out_dir.exists():
        return 0
    removed = 0
    for out_xml in iter_xml_files(out_dir):
        rel = out_xml.relative_to(out_dir)
        in_xml = in_dir / rel
        if in_xml.exists():
            try:
                in_xml.unlink()
                removed += 1
                log(f"  RM INPUT {rel}")
                parent = in_xml.parent
                while parent != in_dir and not any(parent.iterdir()):
                    parent.rmdir()
                    log(f"  RMDIR INPUT {parent.relative_to(in_dir)} (vide)")
                    parent = parent.parent
            except Exception as e:
                log(f"  WARN  Impossible de supprimer {rel} dans input: {e}")
    return removed

# ===== Progress =====
class Progress:
    def __init__(self, total_chars: int):
        self.total_chars = max(1, total_chars)
        self.done_chars = 0
        self.alpha = 0.2
        self.rate = None
    def update(self, delta_chars: int, elapsed: float):
        inst = (delta_chars / elapsed) if elapsed > 0 else 0.0
        self.rate = inst if self.rate is None else self.alpha * inst + (1 - self.alpha) * self.rate
        self.done_chars += delta_chars
    def eta(self) -> str:
        remaining = max(0, self.total_chars - self.done_chars)
        rate = max(1e-6, self.rate or 1e-6)
        return fmt_dur(remaining / rate)
    def pct(self) -> float:
        return 100.0 * self.done_chars / self.total_chars
    def rate_human(self) -> str:
        r = self.rate or 0.0
        return f"{int(r)} chars/s"

# ===== Traduction XML =====
def translate_xml_tree(root: etree._Element, file: Path, prog: Progress):
    for i, text_el in enumerate(root.iter("text"), start=1):
        original = text_el.text or ""
        s = original.strip()
        if not s:
            log(f"    [EMPTY] {file.name} <text #{i}>")
            continue
        if not is_translatable_text(s):
            log(f"    [SKIP-ID] {file.name} <text #{i}> : \"{s}\"")
            continue
        t_start = time.time()
        try:
            translated = translate_text(original, target_lang=TARGET_LANG)
            dt = time.time() - t_start
            text_el.text = translated
            prog.update(len(s), dt)
            log(
                f"    [OK] {file.name} <text #{i}>  "
                f"progress={prog.pct():.1f}% rate={prog.rate_human()} ETA={prog.eta()}"
            )
        except Exception as e:
            dt = time.time() - t_start
            prog.update(len(s), dt)
            parent = text_el.getparent()
            if parent is not None:
                parent.insert(parent.index(text_el), etree.Comment(f"TRANSLATION_ERROR: {e}"))
            log(
                f"    [ERR] {file.name} <text #{i}>: {e}  "
                f"progress={prog.pct():.1f}% ETA={prog.eta()}"
            )

def translate_xml_file(src: Path, dst: Path, file_chars: int, prog: Progress):
    if file_chars <= 0:
        log(f"  SKIP  {src} (0 caractère <text>)")
        return
    log(f"  START {src} (≈{file_chars} chars à traduire)")
    try:
        parser = make_xml_parser()
        tree = etree.parse(str(src), parser)
    except Exception as e:
        log(f"[PARSE_ERR] {src.name}: {e}")
        raise
    t0 = time.time()
    translate_xml_tree(tree.getroot(), src, prog)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(dst), encoding="utf-8", xml_declaration=True, pretty_print=True)
    log(f"  DONE  {dst}  (durée {fmt_dur(time.time()-t0)})")

# ===== Main =====
def main():
    in_dir = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else INPUT_DIR.resolve()
    out_dir = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else OUTPUT_DIR.resolve()

    if not in_dir.exists() or not in_dir.is_dir():
        print(f"Input introuvable ou pas un dossier: {in_dir}")
        sys.exit(1)

    all_input_files = list(iter_xml_files(in_dir))
    log(f"Nombre total de fichiers XML dans input : {len(all_input_files)}")

    log("--- Purge input (déjà présent dans output) ---")
    purged = purge_input_if_output_exists(in_dir, out_dir)
    log(f"Fichiers supprimés de input car présents dans output : {purged}")

    log("--- Pré-scan ---")
    nb_files, total_chars, file_stats = prescan_dir(in_dir, out_dir)
    log(f"Fichiers XML à traiter (existant dans output ignoré) : {nb_files}")
    log(f"Volume total à traduire (caractères dans <text>) : {total_chars}")

    if nb_files == 0 or total_chars == 0:
        log("--- Rien à traduire ---")
        sys.exit(0)

    engine = choose_engine()
    log("--- Démarrage traduction ---")
    log(f"Moteur : {engine.upper()}, Langue cible : {TARGET_LANG}")
    log(f"Input  : {in_dir}")
    log(f"Output : {out_dir}")

    prog = Progress(total_chars=total_chars)
    processed = 0
    errors = 0
    skipped = 0

    for src, (cnt, file_chars) in sorted(file_stats.items(), key=lambda kv: str(kv[0]).lower()):
        rel = src.relative_to(in_dir)
        dst = out_dir / rel
        if dst.exists():
            log(f"  SKIP  {rel} (déjà présent dans output)")
            skipped += 1
            continue
        try:
            translate_xml_file(src, dst, file_chars, prog)
            processed += 1

            try:
                if dst.exists():
                    src.unlink()
                    log(f"  DEL   {rel} (supprimé de input)")
                    parent = src.parent
                    while parent != in_dir and not any(parent.iterdir()):
                        parent.rmdir()
                        log(f"  RMDIR {parent.relative_to(in_dir)} (vide)")
                        parent = parent.parent
                else:
                    log(f"  WARN  Sortie manquante, suppression annulée: {rel}")
            except Exception as e:
                log(f"  WARN  Impossible de supprimer {rel}: {e}")

        except Exception as e:
            log(f"[ERR] Fichier {rel}: {e}")
            errors += 1

    log("--- Terminé ---")
    log(f"Fichiers traduits: {processed}, Skippés: {skipped}, Erreurs: {errors}")
    log(f"Débit moyen: {prog.rate_human()}  Progression: {prog.pct():.1f}%  ETA restant: {prog.eta()}")

    remaining_files = list(iter_xml_files(in_dir))
    log(f"Fichiers restants dans input après traitement : {len(remaining_files)}")

    sys.exit(0 if errors == 0 else 2)

if __name__ == "__main__":
    main()
