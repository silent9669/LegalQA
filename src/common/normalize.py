"""Vietnamese legal text normalization and identifier extraction utilities."""

from __future__ import annotations

import re
import unicodedata
import urllib.parse
from typing import Dict, List, Optional, Set

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None

# Matches legal document identifiers like 90/2017/NĐ-CP, 42/2017/QH14, 100/2015/QH13, 01/2021/TT-BTP, 2475/QĐ-BYT, 01/CĐ-TCT
DOC_NUMBER_PATTERN = re.compile(
    r'\b\d{1,5}/(?:\d{4}/[A-ZĐa-z0-9\-_]+|[A-ZĐa-z0-9]+-[A-ZĐa-z0-9\-]+|\d{4})\b',
    re.IGNORECASE
)

# Individual hierarchy patterns
ARTICLE_PATTERN = re.compile(r'\bĐiều\s+(\d+[a-zA-Z]?)\b', re.IGNORECASE)
CLAUSE_PATTERN = re.compile(r'\bkhoản\s+(\d+[a-zA-Z]?)\b', re.IGNORECASE)
POINT_PATTERN = re.compile(r'\bđiểm\s+([a-zA-Z\d]+)\b', re.IGNORECASE)
YEAR_PATTERN = re.compile(r'\bnăm\s+(\d{4})\b|\b(19\d{2}|20\d{2})\b', re.IGNORECASE)


LEGAL_NAME_MAP = {
    'Bo-luat': 'Bộ luật',
    'Luat': 'Luật',
    'Nghi-dinh': 'Nghị định',
    'Thong-tu': 'Thông tư',
    'Quyet-dinh': 'Quyết định',
    'Nghi-quyet': 'Nghị quyết',
    'Dan-su': 'Dân sự',
    'Hinh-su': 'Hình sự',
    'To-tung': 'Tố tụng',
    'Dat-dai': 'Đất đai',
    'Doanh-nghiep': 'Doanh nghiệp',
    'Lao-dong': 'Lao động',
    'Thuong-mai': 'Thương mại',
    'Nha-o': 'Nhà ở',
    'Giao-thong': 'Giao thông',
    'Xay-dung': 'Xây dựng',
    'Thu-y': 'Thú y',
    'Thue': 'Thuế',
    'De-dieu': 'Đê điều',
    'Hon-nhan': 'Hôn nhân',
    'Gia-dinh': 'Gia đình',
    'Bao-hiem': 'Bảo hiểm',
    'Khoan-chi': 'Khoán chi',
    'Tro-giup-phap-ly': 'Trợ giúp pháp lý',
}


def prettify_doc_title(name: str) -> str:
    """Format a legal document filename slug into human-readable title."""
    if not name or str(name).lower() in ("none", "nan", "null", ""):
        return ""
    unquoted = urllib.parse.unquote(str(name)).strip()
    unquoted = re.sub(r'-\d{4,8}$', '', unquoted)

    m = re.search(r'(Nghi-dinh|Thong-tu|Quyet-dinh|Luat|Bo-luat)-(\d+)-(\d+)-([A-ZĐa-z0-9\-]+)', unquoted, re.I)
    if m:
        type_str = {
            "nghi-dinh": "Nghị định",
            "thong-tu": "Thông tư",
            "quyet-dinh": "Quyết định",
            "luat": "Luật",
            "bo-luat": "Bộ luật",
        }.get(m.group(1).lower(), m.group(1))
        suffix = normalize_legal_number(m.group(4))
        doc_no = f"{m.group(2)}/{m.group(3)}/{suffix}"
        return f"{type_str} {doc_no}"

    res = unquoted
    for k, v in LEGAL_NAME_MAP.items():
        res = re.sub(re.escape(k), v, res, flags=re.I)
    res = res.replace("-", " ")
    res = re.sub(r'\bND CP\b', 'NĐ-CP', res, flags=re.I)
    res = re.sub(r'\bNĐ CP\b', 'NĐ-CP', res, flags=re.I)
    res = re.sub(r'\bTT BTP\b', 'TT-BTP', res, flags=re.I)
    res = re.sub(r'\bQD TTg\b', 'QĐ-TTg', res, flags=re.I)
    res = re.sub(r'\bQĐ TTg\b', 'QĐ-TTg', res, flags=re.I)
    return re.sub(r'\s+', ' ', res).strip()


def remove_accents(input_str: str) -> str:
    """Strip Vietnamese diacritics / accents for robust matching."""
    if not input_str:
        return ""
    nfkd = unicodedata.normalize('NFKD', str(input_str))
    res = ''.join(c for c in nfkd if not unicodedata.combining(c))
    return res.replace('đ', 'd').replace('Đ', 'D')


def slugify_legal_title(text: str) -> str:
    """Convert a legal document title or filename into a canonical normalized slug."""
    if not text:
        return ""
    unquoted = urllib.parse.unquote(str(text))
    # Strip trailing numeric LawNet IDs (e.g. -432844)
    unquoted = re.sub(r'-\d{5,8}$', '', unquoted)
    cleaned = remove_accents(unquoted).lower()
    cleaned = re.sub(r'[^a-z0-9]+', '-', cleaned)
    return cleaned.strip('-')


def clean_legal_text(text: str) -> str:
    """Normalize raw legal text with unicode NFC, clean punctuation and whitespace."""
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r'[\r\t\f\v]', ' ', text)
    text = re.sub(r'[ ]{2,}', ' ', text)
    text = text.replace("\n", " ").strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_question(text: str) -> str:
    """Normalize a QA query: NFC, lowercase, strip punctuation and whitespace."""
    cleaned = clean_legal_text(text).lower()
    cleaned = re.sub(r'[?!.,;:\'"_()]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def normalize_legal_number(doc_num: str) -> str:
    """Canonicalize a legal document number (e.g. 02/2021/tt-btp -> 2/2021/TT-BTP)."""
    if not doc_num:
        return ""
    s = doc_num.strip().upper()
    s = s.replace("ND-CP", "NĐ-CP").replace("QD-TTG", "QĐ-TTg").replace("BLDTBXH", "BLĐTBXH")
    # Normalize leading zero in doc number: 02/2021 -> 2/2021
    m = re.match(r'^0*(\d+)(/.*)$', s)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return s


def extract_canonical_doc_keys(doc_name_or_text: str) -> List[str]:
    """Extract a list of candidate lookup keys from a document name or citation string."""
    if not doc_name_or_text:
        return []
    keys: Set[str] = set()

    slug = slugify_legal_title(doc_name_or_text)
    if slug:
        keys.add(slug)

    # Token-based number and organization extraction from slug
    parts = slug.split('-')
    for i, p in enumerate(parts):
        if p.isdigit() and len(p) == 4 and 1900 <= int(p) <= 2030:
            year = p
            if i > 0 and parts[i-1].isdigit():
                num = parts[i-1]
                # Try extracting org suffix tokens following year (e.g. tt-btp, nd-cp, qd-ttg)
                if i + 1 < len(parts):
                    for org_len in (1, 2, 3):
                        if i + 1 + org_len <= len(parts):
                            org = '-'.join(parts[i+1:i+1+org_len])
                            keys.add(f"{num}/{year}/{org}")
                            keys.add(f"{int(num)}/{year}/{org}")
                keys.add(f"{num}/{year}")
                keys.add(f"{int(num)}/{year}")
        elif i > 1 and parts[i].isdigit() and 1900 <= int(parts[i]) <= 2030:
            # Pattern: Num-Org-Year (e.g. 1450-qd-tct-2021)
            year = parts[i]
            if parts[0].isdigit():
                num = parts[0]
                org = '-'.join(parts[1:i])
                keys.add(f"{num}/{org}")
                keys.add(f"{int(num)}/{org}")
                keys.add(f"{num}/{year}/{org}")

    # Direct regex numbers
    for match in DOC_NUMBER_PATTERN.finditer(doc_name_or_text):
        raw_num = match.group(0).lower()
        keys.add(raw_num)
        raw_slug = slugify_legal_title(raw_num).replace('-', '/')
        keys.add(raw_slug)
        subparts = raw_slug.split('/')
        if len(subparts) >= 2 and subparts[0].isdigit():
            keys.add(f"{int(subparts[0])}/" + "/".join(subparts[1:]))

    return sorted(keys)


def extract_legal_signals(text: str) -> Dict[str, List[str]]:
    """Extract structured statutory signals from legal text or answer."""
    cleaned = clean_legal_text(text)
    doc_nums = [m.group(0).upper() for m in DOC_NUMBER_PATTERN.finditer(cleaned)]
    articles = [m.group(1) for m in ARTICLE_PATTERN.finditer(cleaned)]
    clauses = [m.group(1) for m in CLAUSE_PATTERN.finditer(cleaned)]
    points = [m.group(1) for m in POINT_PATTERN.finditer(cleaned)]

    years = []
    for m in YEAR_PATTERN.finditer(cleaned):
        y = m.group(1) or m.group(2)
        if y:
            years.append(y)

    return {
        "doc_numbers": list(dict.fromkeys(doc_nums)),
        "articles": list(dict.fromkeys(articles)),
        "clauses": list(dict.fromkeys(clauses)),
        "points": list(dict.fromkeys(points)),
        "years": list(dict.fromkeys(years)),
    }


def tokenize_vietnamese(text: str) -> str:
    """Tokenize Vietnamese text with pyvi if available."""
    cleaned = clean_legal_text(text)
    if ViTokenizer is not None:
        return ViTokenizer.tokenize(cleaned)
    return cleaned
