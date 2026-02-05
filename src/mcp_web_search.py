"""
MCP server de búsqueda web para Jim.
SearXNG + RRF Fusion + BM25 Híbrido + CrossEncoder Reranking.

v4 - Production Improvements:
- CrossEncoder BAAI/bge-reranker-v2-m3 (SOTA multilingual)
- Reciprocal Rank Fusion (RRF) para combinar rankings de múltiples motores
- Híbrido BM25 + Semántico con fusión configurable
- Deduplicación agresiva por URL normalizada
- Más contexto scrapeado
- [NEW] Caching con TTL 15 min
- [NEW] Connection pooling con requests.Session
- [NEW] Model warmup al inicio
- [NEW] Health check endpoint
- [NEW] Logging estructurado JSON
"""

import json
import logging
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import md5
from urllib.parse import urlparse, urlunparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from rank_bm25 import BM25Okapi
from requests.adapters import HTTPAdapter

# =========================================================================
# CONFIGURACIÓN
# =========================================================================

SEARXNG_URL = "http://127.0.0.1:8888"

# CrossEncoder model - BAAI/bge-reranker-v2-m3 es SOTA multilingual
# Fallback: cross-encoder/ms-marco-MiniLM-L-6-v2 (más rápido, menos preciso)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_FALLBACK = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Device: "auto" (detecta GPU si hay), "cuda", "cpu"
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "auto")

SCRAPE_TIMEOUT = 5
SCRAPE_MAX_CHARS = 5000  # Más contexto para mejor reranking
SEARXNG_FETCH_COUNT = 50  # Traer más para RRF + rerank + dedup (aumentado de 20)
MAX_WORKERS = 12
RRF_K = 60  # Constante RRF estándar

# Híbrido BM25 + Semántico
HYBRID_ALPHA = 0.3  # 0.3 BM25 + 0.7 Semántico (semántico domina)

# Cache TTL en segundos (15 minutos)
CACHE_TTL = 900

# Stopwords para BM25 (español + inglés básico)
STOPWORDS = {
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "de",
    "del",
    "en",
    "y",
    "o",
    "a",
    "al",
    "que",
    "es",
    "son",
    "para",
    "por",
    "con",
    "se",
    "su",
    "the",
    "a",
    "an",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "como",
    "más",
    "pero",
    "si",
    "no",
    "ya",
    "sin",
    "sobre",
    "entre",
    "cuando",
}

# =========================================================================
# LOGGING ESTRUCTURADO
# =========================================================================

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("mcp-web-search")


def _log_search(
    query: str, categories: str, latency: float, cache_hit: bool, results: int
):
    """Log estructurado JSON para cada búsqueda."""
    logger.info(
        json.dumps(
            {
                "event": "search",
                "query": query[:100],
                "categories": categories,
                "latency_ms": round(latency * 1000),
                "cache_hit": cache_hit,
                "results": results,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
    )


# =========================================================================
# CACHE CON TTL
# =========================================================================

_cache: dict[str, tuple[str, float]] = {}


def _cache_key(query: str, categories: str, language: str, time_range: str = "") -> str:
    """Genera cache key único basado en parámetros de búsqueda."""
    return md5(f"{query}|{categories}|{language}|{time_range}".encode()).hexdigest()


def _get_cached(key: str) -> str | None:
    """Obtiene resultado cacheado si existe y no expiró."""
    if key in _cache:
        result, timestamp = _cache[key]
        if time.time() - timestamp < CACHE_TTL:
            return result
        del _cache[key]
    return None


def _set_cache(key: str, result: str):
    """Guarda resultado en cache con timestamp."""
    _cache[key] = (result, time.time())


def _cache_stats() -> dict:
    """Estadísticas del cache."""
    now = time.time()
    valid = sum(1 for _, (_, ts) in _cache.items() if now - ts < CACHE_TTL)
    return {
        "total_entries": len(_cache),
        "valid_entries": valid,
        "ttl_seconds": CACHE_TTL,
    }


# =========================================================================
# CONNECTION POOLING
# =========================================================================

_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """Obtiene Session con connection pooling configurado."""
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=2,
        )
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
        _session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            }
        )
    return _session


# =========================================================================
# LAZY MODELS
# =========================================================================

_reranker = None


def _get_reranker():
    """Carga lazy del CrossEncoder con fallback."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        # Determinar device: auto detecta GPU, o forzar cuda/cpu
        device = None if RERANKER_DEVICE == "auto" else RERANKER_DEVICE

        try:
            _reranker = CrossEncoder(
                RERANKER_MODEL,
                trust_remote_code=True,
                device=device,
            )
        except Exception:
            # Fallback a modelo más ligero
            _reranker = CrossEncoder(RERANKER_FALLBACK, device=device)
    return _reranker


def _normalize_url(url: str) -> str:
    """Normaliza URL para deduplicación."""
    try:
        parsed = urlparse(url.lower())
        # Remover www., trailing slash, fragments, algunos params
        netloc = parsed.netloc.replace("www.", "")
        path = parsed.path.rstrip("/")
        # Ignorar query params para dedup (pueden variar)
        return urlunparse(("", netloc, path, "", "", ""))
    except Exception:
        return url.lower()


def _tokenize(text: str) -> list[str]:
    """Tokeniza texto para BM25."""
    text = text.lower()
    text = re.sub(r"[^\w\sáéíóúñü]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def _normalize_scores(scores: list[float]) -> list[float]:
    """Normaliza scores a [0, 1]."""
    if not scores:
        return []
    arr = np.array(scores)
    min_s, max_s = arr.min(), arr.max()
    if max_s - min_s < 1e-9:
        return [0.5] * len(scores)
    return ((arr - min_s) / (max_s - min_s)).tolist()


def _rerank_with_crossencoder(query: str, documents: list[str]) -> list[float]:
    """Re-rankea documentos usando CrossEncoder."""
    if not documents:
        return []
    reranker = _get_reranker()
    pairs = [(query, doc) for doc in documents]
    scores = reranker.predict(pairs)
    return scores.tolist() if hasattr(scores, "tolist") else list(scores)


def _bm25_scores(query: str, documents: list[str]) -> list[float]:
    """Calcula scores BM25 para documentos."""
    if not documents:
        return []

    tokenized_docs = [_tokenize(doc) for doc in documents]
    tokenized_query = _tokenize(query)

    # Filtrar docs vacíos después de tokenizar
    if not any(tokenized_docs) or not tokenized_query:
        return [0.0] * len(documents)

    try:
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(tokenized_query)
        return scores.tolist() if hasattr(scores, "tolist") else list(scores)
    except Exception:
        return [0.0] * len(documents)


def _hybrid_rerank(
    query: str, documents: list[str], alpha: float = HYBRID_ALPHA
) -> list[float]:
    """
    Híbrido BM25 + CrossEncoder.
    alpha: peso de BM25 (1-alpha para semántico)
    """
    if not documents:
        return []

    # BM25 scores (lexical)
    bm25_raw = _bm25_scores(query, documents)
    bm25_norm = _normalize_scores(bm25_raw)

    # CrossEncoder scores (semantic)
    semantic_raw = _rerank_with_crossencoder(query, documents)
    semantic_norm = _normalize_scores(semantic_raw)

    # Fusión lineal
    hybrid = []
    for i in range(len(documents)):
        bm25_s = bm25_norm[i] if i < len(bm25_norm) else 0
        sem_s = semantic_norm[i] if i < len(semantic_norm) else 0
        hybrid.append(alpha * bm25_s + (1 - alpha) * sem_s)

    return hybrid


def _reciprocal_rank_fusion(
    results_by_engine: dict[str, list[str]], k: int = RRF_K
) -> dict[str, float]:
    """
    Combina rankings de múltiples motores usando RRF.
    Formula: score(d) = sum(1 / (k + rank_i(d))) para cada motor i
    """
    scores = defaultdict(float)
    for engine, urls in results_by_engine.items():
        for rank, url in enumerate(urls, start=1):
            norm_url = _normalize_url(url)
            scores[norm_url] += 1.0 / (k + rank)
    return dict(scores)


def _scrape_page(url: str) -> str:
    """Scrapea una página y extrae texto limpio."""
    try:
        resp = _get_session().get(url, timeout=SCRAPE_TIMEOUT)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type:
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")

        # Eliminar elementos no-contenido
        for tag in soup.find_all(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "aside",
                "iframe",
                "noscript",
                "form",
                "svg",
            ]
        ):
            tag.decompose()

        # Priorizar contenido principal
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find(class_=re.compile(r"content|article|post"))
        )
        if main_content:
            text = main_content.get_text(separator=" ", strip=True)
        else:
            text = soup.get_text(separator=" ", strip=True)

        text = " ".join(text.split())
        return text[:SCRAPE_MAX_CHARS] if text else ""
    except Exception:
        return ""


def _search_searxng(
    query: str,
    categories: str = "general",
    time_range: str = "",
    language: str = "",
    count: int = SEARXNG_FETCH_COUNT,
) -> list[dict]:
    """Busca en SearXNG y devuelve resultados crudos."""
    params = {
        "q": query,
        "format": "json",
        "categories": categories,
    }
    if time_range:
        params["time_range"] = time_range
    if language:
        params["language"] = language

    try:
        resp = _get_session().get(f"{SEARXNG_URL}/search", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        return results[:count]
    except Exception as e:
        return [{"error": str(e)}]


def _search_and_rerank(
    query: str,
    max_results: int,
    categories: str,
    time_range: str = "",
    language: str = "",
) -> str:
    """
    Pipeline completo:
    1. Cache check → devolver si existe
    2. SearXNG → obtener resultados de múltiples motores
    3. Deduplicación por URL normalizada
    4. RRF Fusion → combinar rankings por motor
    5. Scrape → enriquecer con contenido
    6. Híbrido BM25 + CrossEncoder → reranking preciso
    7. Cache store → guardar resultado
    """
    start_time = time.time()

    # 0. Cache check
    cache_key = _cache_key(query, categories, language, time_range)
    cached = _get_cached(cache_key)
    if cached:
        _log_search(
            query, categories, time.time() - start_time, cache_hit=True, results=-1
        )
        return cached

    # 1. Buscar en SearXNG
    raw_results = _search_searxng(query, categories, time_range, language)

    if not raw_results:
        return f"Sin resultados para '{query}'."

    if "error" in raw_results[0]:
        return f"Error buscando en SearXNG: {raw_results[0]['error']}"

    # 2. Agrupar por motor para RRF + deduplicar
    results_by_engine = defaultdict(list)
    url_to_result = {}  # norm_url -> result data
    original_urls = {}  # norm_url -> original_url

    for r in raw_results:
        url = r.get("url", "")
        if not url:
            continue

        norm_url = _normalize_url(url)
        engine = r.get("engine", "unknown")
        results_by_engine[engine].append(url)

        if norm_url not in url_to_result:
            url_to_result[norm_url] = {
                "url": url,  # Mantener URL original
                "title": r.get("title", ""),
                "snippet": r.get("content", ""),
                "engines": [engine],
            }
            original_urls[norm_url] = url
        else:
            # Agregar motor a resultado existente
            if engine not in url_to_result[norm_url]["engines"]:
                url_to_result[norm_url]["engines"].append(engine)

    # 3. RRF Fusion
    rrf_scores = _reciprocal_rank_fusion(results_by_engine)

    # Ordenar por RRF y tomar top candidates
    sorted_urls = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    # Tomar más candidatos para compensar posibles fallos de scraping
    top_norm_urls = [norm_url for norm_url, _ in sorted_urls[: max_results * 3]]

    # 4. Scraping paralelo
    enriched = []
    num_workers = min(MAX_WORKERS, len(top_norm_urls))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_url = {
            executor.submit(_scrape_page, original_urls[norm_url]): norm_url
            for norm_url in top_norm_urls
            if norm_url in original_urls
        }

        for future in as_completed(future_to_url):
            norm_url = future_to_url[future]
            page_text = future.result()
            result = url_to_result[norm_url]

            full_text = f"{result['title']}. {result['snippet']}"
            if page_text:
                full_text += f" {page_text}"

            enriched.append(
                {
                    "title": result["title"],
                    "url": result["url"],
                    "snippet": result["snippet"],
                    "page_text": page_text[:1000] if page_text else "",
                    "full_text": full_text,
                    "engines": result["engines"],
                    "rrf_score": rrf_scores.get(norm_url, 0),
                    "engine_count": len(result["engines"]),
                }
            )

    if not enriched:
        return f"No se pudo obtener contenido para '{query}'."

    # 5. Hybrid reranking (BM25 + CrossEncoder)
    texts = [e["full_text"] for e in enriched]
    hybrid_scores = _hybrid_rerank(query, texts)

    for i, item in enumerate(enriched):
        item["hybrid_score"] = hybrid_scores[i] if i < len(hybrid_scores) else 0
        # Bonus por aparecer en múltiples motores
        engine_bonus = 0.05 * (item["engine_count"] - 1)
        item["final_score"] = min(1.0, item["hybrid_score"] + engine_bonus)

    # 6. Ordenar por score final
    enriched.sort(key=lambda x: x["final_score"], reverse=True)

    # 7. Formatear output
    top = enriched[:max_results]
    engine_list = list(results_by_engine.keys())
    lines = [
        f"Resultados para '{query}' ({len(top)} de {len(enriched)}):",
        f"Pipeline: RRF + BM25 híbrido + CrossEncoder | Motores: {', '.join(engine_list[:5])}",
        "",
    ]

    for i, item in enumerate(top, 1):
        score_pct = item["final_score"] * 100
        engines_str = ", ".join(item["engines"])
        lines.append(f"{i}. [{score_pct:.0f}%] {item['title']}")
        lines.append(f"   URL: {item['url']}")
        if item["snippet"]:
            lines.append(f"   {item['snippet'][:350]}")
        if item["page_text"] and item["page_text"] != item["snippet"]:
            lines.append(f"   Extracto: {item['page_text'][:350]}...")
        lines.append(f"   Motores: {engines_str}")
        lines.append("")

    result = "\n".join(lines)

    # 8. Cache store + log
    _set_cache(cache_key, result)
    _log_search(
        query, categories, time.time() - start_time, cache_hit=False, results=len(top)
    )

    return result


# =========================================================================
# MCP TOOLS
# =========================================================================

mcp = FastMCP("web-search")


@mcp.tool()
def web_search(
    query: str,
    max_results: int = 8,
    categories: str = "general",
    language: str = "",
) -> str:
    """Búsqueda web avanzada con RRF + BM25 híbrido + CrossEncoder reranking.

    Pipeline: SearXNG → Dedup → RRF Fusion → Scraping → Híbrido (BM25 + CrossEncoder)

    Más preciso que búsqueda simple porque:
    - RRF combina rankings de múltiples motores (consenso = mejor)
    - BM25 captura coincidencias léxicas exactas
    - CrossEncoder evalúa semántica query+documento
    - Bonus para resultados que aparecen en múltiples motores

    Args:
        query: Qué buscar (lenguaje natural, ej: "heladerias en palermo buenos aires")
        max_results: Cantidad de resultados (default 8)
        categories: Categorías SearXNG: general, news, science, it, images, videos
        language: Código idioma/región (ej: "es-AR", "en-US"). Default: auto.

    Returns:
        Resultados ordenados por relevancia híbrida.
    """
    return _search_and_rerank(query, max_results, categories, language=language)


@mcp.tool()
def web_search_news(
    query: str,
    max_results: int = 8,
    timelimit: str = "week",
    language: str = "",
) -> str:
    """Busca noticias recientes con reranking avanzado.

    Usa motores de noticias (Google News, Bing News, etc).
    Ideal para eventos actuales, mercados, noticias locales.

    Args:
        query: Qué buscar (ej: "dólar blue hoy", "elecciones argentina")
        max_results: Cantidad de resultados (default 8)
        timelimit: Rango: "day", "week", "month", "year" (default "week")
        language: Código idioma/región (ej: "es-AR"). Default: auto.

    Returns:
        Noticias ordenadas por relevancia.
    """
    return _search_and_rerank(
        query, max_results, "news", time_range=timelimit, language=language
    )


@mcp.tool()
def health_check() -> str:
    """Verifica estado del servidor MCP y sus dependencias.

    Útil para monitoring y debugging. Verifica:
    - Conexión a SearXNG
    - Estado del modelo de reranking
    - Estadísticas del cache

    Returns:
        Estado del servidor y sus componentes.
    """
    checks = {
        "searxng": False,
        "searxng_url": SEARXNG_URL,
        "reranker_loaded": _reranker is not None,
        "reranker_model": RERANKER_MODEL if _reranker else "not loaded",
        "cache": _cache_stats(),
        "session_active": _session is not None,
    }

    # Test SearXNG
    try:
        resp = _get_session().get(f"{SEARXNG_URL}/healthz", timeout=2)
        checks["searxng"] = resp.status_code == 200
    except Exception as e:
        checks["searxng_error"] = str(e)

    status = "healthy" if checks["searxng"] else "degraded"
    lines = [f"Status: {status}", ""]
    for key, value in checks.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for k, v in value.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"{key}: {value}")

    return "\n".join(lines)


# =========================================================================
# WARMUP
# =========================================================================


def _warmup():
    """Pre-carga el modelo CrossEncoder para evitar latencia en primera query."""
    logger.info(
        json.dumps({"event": "warmup", "status": "starting", "model": RERANKER_MODEL})
    )
    try:
        _get_reranker()
        # Dummy inference para compilar/optimizar
        _rerank_with_crossencoder("warmup test query", ["warmup test document"])
        logger.info(json.dumps({"event": "warmup", "status": "complete"}))
    except Exception as e:
        logger.error(
            json.dumps({"event": "warmup", "status": "failed", "error": str(e)})
        )


if __name__ == "__main__":
    _warmup()
    mcp.run()
