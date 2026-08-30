import re
import unicodedata

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None

DOC_NUMBER_PATTERN = re.compile(
    r'\b\d{1,5}/(?:\d{4}(?:/[A-ZĐa-z\-]+)?|(?:[A-ZĐa-z]+-[A-ZĐa-z]+))\b',
    re.IGNORECASE
)
ARTICLE_PATTERN = re.compile(r'\bĐiều\s+(\d+[a-zA-Z]?)\b', re.IGNORECASE)
CLAUSE_PATTERN = re.compile(r'\bkhoản\s+(\d+[a-zA-Z]?)\b', re.IGNORECASE)
POINT_PATTERN = re.compile(r'\bđiểm\s+([a-zA-Z\d]+)\b', re.IGNORECASE)
YEAR_PATTERN = re.compile(r'\bnăm\s+(\d{4})\b|\b(19\d{2}|20\d{2})\b', re.IGNORECASE)

def clean_legal_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r'[\r\t\f\v]', ' ', text)
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r'[ ]{2,}', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = text.replace("\n", " ").strip()
    text = re.sub(r'\s+', ' ', text)
    return text

LEGAL_NAME_MAP = {
    'Bo-luat': 'Bộ luật',
    'Luat': 'Luật',
    'Nghi-dinh': 'Nghị định',
    'Thong-tu': 'Thông tư',
    'Quyet-dinh': 'Quyết định',
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
    'Tro-giup-phap-ly': 'Trợ giúp pháp lý'
}

def prettify_doc_title(name: str) -> str:
    if not name or str(name).lower() in ("none", "nan", "null", ""):
        return ""
    name = str(name).strip()
    name = re.sub(r'-\d{4,8}$', '', name)
    m = re.search(r'(Nghi-dinh|Thong-tu|Quyet-dinh|Luat|Bo-luat)-(\d+)-(\d+)-([A-ZĐa-z]+-[A-ZĐa-z]+|[A-ZĐa-z]+)', name, re.I)
    if m:
        type_str = {
            "nghi-dinh": "Nghị định",
            "thong-tu": "Thông tư",
            "quyet-dinh": "Quyết định",
            "luat": "Luật",
            "bo-luat": "Bộ luật"
        }.get(m.group(1).lower(), m.group(1))
        suffix = m.group(4).upper().replace("ND-CP", "NĐ-CP").replace("QD-TTG", "QĐ-TTg").replace("BLDTBXH", "BLĐTBXH")
        doc_no = f"{m.group(2)}/{m.group(3)}/{suffix}"
        return f"{type_str} {doc_no}"

    for k, v in LEGAL_NAME_MAP.items():
        name = re.sub(re.escape(k), v, name, flags=re.I)
    name = name.replace("-", " ")
    name = re.sub(r'\bND CP\b', 'NĐ-CP', name, flags=re.I)
    name = re.sub(r'\bTT BTP\b', 'TT-BTP', name, flags=re.I)
    name = re.sub(r'\bQD TTg\b', 'QĐ-TTg', name, flags=re.I)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def normalize_question(text: str) -> str:
    cleaned = clean_legal_text(text).lower()
    cleaned = re.sub(r'[?!.,;:\'"_]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def extract_legal_signals(text: str) -> dict:
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
        "years": list(dict.fromkeys(years))
    }

def tokenize_vietnamese(text: str) -> str:
    cleaned = clean_legal_text(text)
    if ViTokenizer is not None:
        return ViTokenizer.tokenize(cleaned)
    return cleaned
