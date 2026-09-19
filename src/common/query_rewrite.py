"""Query rewriting for retrieval: acronym expansion + weighted RRF policy.

Inherited from the v10 retrieval lab. Expansion APPENDS the long form (the
original token is kept) at most once per key per query. The weighted policy
up-weights BM25 on queries carrying explicit legal references; plain queries
keep uniform weights. Both are opt-in: default fusion behavior is unchanged.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

ACRONYMS: Dict[str, str] = {
    "vphc": "vi phạm hành chính",
    "xphc": "xử phạt vi phạm hành chính",
    "đkkd": "đăng ký kinh doanh",
    "đkdn": "đăng ký doanh nghiệp",
    "gcn": "giấy chứng nhận",
    "gcnqsdđ": "giấy chứng nhận quyền sử dụng đất",
    "nlđ": "người lao động",
    "nsdlđ": "người sử dụng lao động",
    "hđlđ": "hợp đồng lao động",
    "hđtv": "hội đồng thành viên",
    "hđqt": "hội đồng quản trị",
    "csgt": "cảnh sát giao thông",
    "gpkd": "giấy phép kinh doanh",
    "gplx": "giấy phép lái xe",
    "bhxh": "bảo hiểm xã hội",
    "bhyt": "bảo hiểm y tế",
    "bhtn": "bảo hiểm thất nghiệp",
    "ubnd": "ủy ban nhân dân",
    "hđnd": "hội đồng nhân dân",
    "tnhh": "trách nhiệm hữu hạn",
    "dnnn": "doanh nghiệp nhà nước",
    "atgt": "an toàn giao thông",
    "pccc": "phòng cháy chữa cháy",
    "vsattp": "vệ sinh an toàn thực phẩm",
    "tttt": "thông tin truyền thông",
    "qsdđ": "quyền sử dụng đất",
    "sxkd": "sản xuất kinh doanh",
}

_ACR_RE = re.compile(r"\b(" + "|".join(sorted(ACRONYMS, key=len, reverse=True)) + r")\b", re.I)

# Query carries an explicit legal reference (điều/khoản/điểm, doc number...).
_LEXQ = re.compile(
    r"(?:\bđiều\s*\d|\bkhoản\s*\d|\bđiểm\s*[a-zđ]\b|\d{1,4}\s*/\s*\d{2,4}"
    r"|\bnghị\s*định\b|\bthông\s*tư\b|\bluật\b)",
    re.I,
)

def expand_acronyms(query: str) -> str:
    """Append long forms for known Vietnamese legal acronyms (NFC-safe)."""
    s = unicodedata.normalize("NFC", str(query or ""))
    seen: set = set()

    def _replace(match: re.Match) -> str:
        key = match.group(1).lower()
        if key in seen:
            return match.group(0)
        seen.add(key)
        return f"{match.group(0)} {ACRONYMS[key]}"

    return _ACR_RE.sub(_replace, s)


def has_legal_reference(query: str) -> bool:
    """Detect explicit legal references (điều/khoản/điểm, doc numbers)."""
    return bool(_LEXQ.search(unicodedata.normalize("NFC", str(query or ""))))


def rewrite_query_for_retrieval(query: str, use_acronyms: bool = False) -> str:
    """Rewritten retrieval query; identity when the experiment is off."""
    if not use_acronyms:
        return str(query or "")
    return expand_acronyms(query)
