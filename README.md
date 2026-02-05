# MCP Web Search

Servidor MCP de búsqueda web con reranking avanzado usando SearXNG como backend.

## Features

- **SearXNG** como metabuscador (40+ motores de búsqueda)
- **CrossEncoder** (BAAI/bge-reranker-v2-m3) para reranking semántico
- **BM25** para matching léxico
- **RRF (Reciprocal Rank Fusion)** para combinar rankings de múltiples motores
- **Híbrido BM25 + Semántico** con fusión configurable (30% léxico, 70% semántico)
- **Caching** con TTL de 15 minutos
- **Connection pooling** para mejor performance
- **Scraping** de contenido para enriquecer resultados
- **Warmup** del modelo al inicio para evitar latencia en primera query

## Requisitos

- Python 3.11+
- Docker (para SearXNG)
- ~3GB de espacio (modelo CrossEncoder)

## Estructura del proyecto

```
.
├── src/
│   └── mcp_web_search.py      # Servidor MCP principal (605 líneas)
├── docker/
│   └── searxng/
│       ├── docker-compose.yml # Config Docker para SearXNG
│       └── settings.yml       # Configuración de motores (266 líneas)
├── requirements.txt           # Dependencias Python
└── README.md
```

## Instalación

### 1. Clonar/copiar el proyecto

```bash
git clone <repo> ~/mcp/web-search
cd ~/mcp/web-search
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac/WSL
# o: .\venv\Scripts\activate  # Windows PowerShell

pip install -r requirements.txt
```

**Dependencias:**
- `mcp>=1.0.0` - Framework MCP
- `requests>=2.28.0` - HTTP client
- `beautifulsoup4>=4.12.0` - Scraping
- `sentence-transformers>=2.2.0` - CrossEncoder reranking
- `rank-bm25>=0.2.2` - BM25 scoring
- `numpy>=1.24.0` - Operaciones numéricas

### 3. Levantar SearXNG

```bash
cd docker/searxng
docker compose up -d
```

Verificar que funcione:
```bash
curl "http://127.0.0.1:8888/search?q=test&format=json" | jq '.results | length'
# Debería devolver un número > 0
```

### 4. Probar el MCP

```bash
cd ~/mcp/web-search
./venv/bin/python src/mcp_web_search.py
```

La primera ejecución descarga el modelo CrossEncoder (~1.5GB de HuggingFace).

## Configuración en Claude Code

Agregar a `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "web-search": {
      "command": "/home/USER/mcp/web-search/venv/bin/python",
      "args": ["/home/USER/mcp/web-search/src/mcp_web_search.py"]
    }
  }
}
```

Reemplazar `USER` con tu usuario.

## Tools disponibles

### `web_search`

Búsqueda web general con reranking avanzado.

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `query` | string | requerido | Qué buscar |
| `max_results` | int | 8 | Cantidad de resultados (1-20) |
| `categories` | string | "general" | Categoría de búsqueda |
| `language` | string | auto | Código idioma (ej: "es-AR", "en-US") |

**Ejemplo:**
```
web_search("python asyncio tutorial", max_results=5, categories="it")
```

### `web_search_news`

Búsqueda de noticias recientes con filtro temporal.

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `query` | string | requerido | Qué buscar |
| `max_results` | int | 8 | Cantidad de resultados |
| `time_range` | string | "week" | day, week, month, year |
| `language` | string | auto | Código idioma |

### `health_check`

Verifica estado del servidor y dependencias.

**Retorna:**
- Estado de SearXNG
- Estado del reranker (cargado/no cargado)
- Estadísticas de cache
- Estado de la sesión HTTP

## Categorías de SearXNG

| Categoría | Motores |
|-----------|---------|
| `general` | google, bing, duckduckgo, brave, yahoo, qwant, wikipedia, google scholar, openlibrary |
| `it` | google, bing, duckduckgo, brave, github, gitlab, npm, pypi, crates.io, hacker news, lemmy |
| `news` | google news, bing news, yahoo news, brave news, qwant news, duckduckgo news, reuters |
| `science` | google scholar, arxiv, semantic scholar, pubmed, crossref |
| `videos` | youtube, vimeo, dailymotion, google videos, bing videos |
| `images` | google images, bing images, duckduckgo images, unsplash |
| `social_media` | reddit, lemmy, mastodon |

## Pipeline de búsqueda

```
Query del usuario
       ↓
┌──────────────────┐
│    SearXNG       │  → Consulta 40+ motores en paralelo
│  (50 resultados) │
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Deduplicación   │  → Normaliza URLs, elimina duplicados
└────────┬─────────┘
         ↓
┌──────────────────┐
│   RRF Fusion     │  → Combina rankings de múltiples motores
│    (k=60)        │     score = Σ 1/(k + rank_i)
└────────┬─────────┘
         ↓
┌──────────────────┐
│    Scraping      │  → Extrae contenido de cada URL (paralelo)
│  (5s timeout)    │     hasta 5000 chars por página
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Hybrid Rerank   │  → 30% BM25 (léxico) + 70% CrossEncoder (semántico)
│ (BM25 + CE)      │
└────────┬─────────┘
         ↓
   Top N resultados
```

## Configuración avanzada

### Variables de entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `RERANKER_DEVICE` | `auto` | Device para CrossEncoder: `auto`, `cuda`, `cpu` |

**Ejemplo para forzar GPU:**
```bash
RERANKER_DEVICE=cuda ./venv/bin/python src/mcp_web_search.py
```

### GPU Support (AMD ROCm)

El reranking es **33x más rápido** en GPU vs CPU.

| GPU | Reranking 30 docs |
|-----|-------------------|
| AMD RX 9060 XT | 90ms |
| CPU | 3000ms |

#### Requisitos AMD ROCm (Linux/WSL2)

1. **Windows (para WSL2):** AMD Adrenalin 26.1.1+ con soporte WSL2
2. **WSL2/Linux:** ROCm 7.2+

#### Instalación ROCm 7.2 en WSL2

```bash
# Descargar instalador
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
sudo apt install ./amdgpu-install_7.2.70200-1_all.deb

# Instalar ROCm para WSL
sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms

# Verificar
rocminfo | grep "Marketing Name"
```

#### PyTorch con ROCm 7.2

```bash
cd ~/mcp/web-search
source venv/bin/activate

# Instalar PyTorch ROCm 7.2 desde AMD repo
pip install 'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl'
pip install 'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl' --no-deps
pip install 'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl' --no-deps

# Verificar
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

#### GPUs AMD soportadas (ROCm 7.2)

- RX 9060, RX 9060 XT, RX 9070, RX 9070 XT (RDNA4)
- RX 7900 XTX, RX 7900 XT, RX 7800 XT, RX 7600 (RDNA3)

### GPU Support (NVIDIA CUDA)

Para NVIDIA, PyTorch detecta CUDA automáticamente. Solo instalar drivers NVIDIA y CUDA toolkit.

### Parámetros en `src/mcp_web_search.py`

```python
# SearXNG
SEARXNG_URL = "http://127.0.0.1:8888"
SEARXNG_FETCH_COUNT = 50  # Resultados iniciales a traer

# Reranking
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # SOTA multilingual
RERANKER_FALLBACK = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Más rápido

# Híbrido
HYBRID_ALPHA = 0.3  # 0.3 BM25 + 0.7 Semántico

# Scraping
SCRAPE_TIMEOUT = 5  # segundos
SCRAPE_MAX_CHARS = 5000  # por página
MAX_WORKERS = 12  # threads paralelos

# Cache
CACHE_TTL = 900  # 15 minutos

# RRF
RRF_K = 60  # constante estándar
```

### Agregar/modificar motores en SearXNG

Editar `docker/searxng/settings.yml`:

```yaml
engines:
  - name: nombre_motor
    engine: tipo_engine
    shortcut: xx
    disabled: false
    categories: [general, it]  # categorías donde aparece
    timeout: 6.0  # opcional
```

Reiniciar después de cambios:
```bash
cd docker/searxng
docker compose restart
```

Ver motores disponibles: https://docs.searxng.org/user/configured_engines.html

## Troubleshooting

### SearXNG no responde

```bash
cd docker/searxng
docker compose logs -f  # Ver logs
docker compose restart  # Reiniciar
docker compose down && docker compose up -d  # Recrear
```

### Modelo no carga

El modelo se descarga de HuggingFace la primera vez (~1.5GB). Verificar:
- Conexión a internet
- Espacio en disco (~3GB necesarios)
- Timeout (puede tardar varios minutos la primera vez)

### Brave rate limiting

Brave usa scraping y puede dar "too many requests". Opciones:
1. Dejarlo así - otros motores compensan
2. Desactivarlo en `settings.yml`: `disabled: true`

El motor se suspende temporalmente pero el pipeline continúa con los demás.

### Categoría no devuelve resultados

Verificar que los motores de esa categoría estén habilitados en `docker/searxng/settings.yml`.

### Cache

El cache tiene TTL de 15 minutos. Para forzar búsqueda fresca, esperar o reiniciar el MCP.

Ver estadísticas de cache:
```python
health_check()  # Muestra cache entries y TTL
```

## Logging

El servidor genera logs JSON estructurados:

```json
{
  "event": "search",
  "query": "python asyncio",
  "categories": "general",
  "latency_ms": 2340,
  "cache_hit": false,
  "results": 8,
  "timestamp": "2026-02-05T05:30:00Z"
}
```

## Performance

| Métrica | Valor típico |
|---------|--------------|
| Primera query (carga modelo) | 10-30s |
| Query con cache miss | 2-5s |
| Query con cache hit | <100ms |
| Memoria (modelo cargado) | ~2GB |

## Notas técnicas

### CrossEncoder sin Sigmoid

El modelo `bge-reranker-v2-m3` produce scores que se normalizan con min-max en `_normalize_scores()`. No usamos `Sigmoid()` como activation function porque comprime los scores y pierde resolución.

### RRF vs promedio simple

RRF (Reciprocal Rank Fusion) es más robusto que promediar rankings porque:
- Penaliza menos los rankings bajos
- Maneja mejor motores con diferente cantidad de resultados
- Fórmula: `score = Σ 1/(k + rank_i)` donde k=60

### Deduplicación

URLs se normalizan antes de deduplicar:
- Lowercase
- Remueve `www.`
- Remueve trailing slash
- Ignora fragments y query params

## Licencia

MIT
