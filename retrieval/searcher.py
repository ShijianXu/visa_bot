"""Web search module – targets official government and embassy sources.

Search priority:
  1. Tavily (if TAVILY_API_KEY is set) – purpose-built for RAG, returns
     pre-cleaned page content so many pages don't need scraping.
  2. DuckDuckGo – free fallback with no rate limits in practice.

In both cases, curated OFFICIAL_SOURCES for the destination are prepended to
the result list so they are always scraped first.
"""

import config
from ddgs import DDGS

# Domains considered authoritative for visa info
_OFFICIAL_DOMAINS = (
    ".gov", ".gob", ".gouv", ".gc.ca", ".gov.uk", ".gov.au",
    ".admin.ch", ".europa.eu", "embassy", "consulate", "immigration",
    "mfa.", "mofa.", "mfaic.", "travel.state.gov", "itamaraty",
    "vfsglobal", "tlscontact", "blsintl",
)

# Curated official visa portals per destination country.
# These are always prepended to search results so they are scraped first,
# regardless of whether the search engine surfaces them.
OFFICIAL_SOURCES: dict[str, list[str]] = {
    "japan":           ["https://www.mofa.go.jp/j_info/visit/visa/index.html"],
    "uk":              ["https://www.gov.uk/check-uk-visa"],
    "united kingdom":  ["https://www.gov.uk/check-uk-visa"],
    "usa":             ["https://travel.state.gov/content/travel/en/us-visas.html"],
    "united states":   ["https://travel.state.gov/content/travel/en/us-visas.html"],
    "canada":          ["https://www.canada.ca/en/immigration-refugees-citizenship/services/visit-canada.html"],
    "australia":       ["https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-finder"],
    "new zealand":     ["https://www.immigration.govt.nz/new-zealand-visas"],
    "france":          ["https://france-visas.gouv.fr/en/web/france-visas"],
    "germany":         ["https://www.auswaertiges-amt.de/en/visa-service"],
    "italy":           ["https://vistoperitalia.esteri.it/home/en"],
    "spain":           ["https://www.exteriores.gob.es/en/ServiciosAlCiudadano/Paginas/Solicitud-de-visado.aspx"],
    "portugal":        ["https://www.vistos.mne.pt/en/"],
    "netherlands":     ["https://www.netherlandsworldwide.nl/visas-for-the-netherlands"],
    "belgium":         ["https://diplomatie.belgium.be/en/services/travelling_abroad"],
    "switzerland":     ["https://www.sem.admin.ch/sem/en/home/themen/einreise.html"],
    "austria":         ["https://www.bmeia.gv.at/en/travel-stay/entry-and-residence/"],
    "sweden":          ["https://www.migrationsverket.se/English/Private-individuals/Visiting-Sweden.html"],
    "norway":          ["https://www.udi.no/en/want-to-apply/visit/"],
    "denmark":         ["https://www.nyidanmark.dk/en-GB"],
    "finland":         ["https://migri.fi/en/visiting-finland"],
    "greece":          ["https://www.mfa.gr/en/visas/"],
    "poland":          ["https://www.gov.pl/web/mfa/visa"],
    "czech republic":  ["https://www.mzv.cz/jnp/en/information_for_aliens/short_stay_visa/index.html"],
    "hungary":         ["https://bmbah.hu/index.php?lang=en"],
    "romania":         ["https://evisa.mae.ro/"],
    "croatia":         ["https://mup.gov.hr/aliens-281621/281621"],
    "china":           ["https://www.visaforchina.cn/"],
    "south korea":     ["https://www.visa.go.kr/openPage.do?MENU_ID=10101"],
    "singapore":       ["https://www.ica.gov.sg/enter-transit-depart/entering-singapore"],
    "thailand":        ["https://www.thaievisa.go.th/"],
    "vietnam":         ["https://evisa.xuatnhapcanh.gov.vn/"],
    "malaysia":        ["https://www.imi.gov.my/"],
    "indonesia":       ["https://evisa.imigrasi.go.id/"],
    "philippines":     ["https://evisa.bureau.immigration.gov.ph/"],
    "india":           ["https://indianvisaonline.gov.in/"],
    "sri lanka":       ["https://eta.gov.lk/"],
    "nepal":           ["https://www.nepalimmigration.gov.np/"],
    "myanmar":         ["https://evisa.moip.gov.mm/"],
    "cambodia":        ["https://www.evisa.gov.kh/"],
    "laos":            ["https://laoevisa.gov.la/"],
    "taiwan":          ["https://www.boca.gov.tw/mp-2.html"],
    "mongolia":        ["https://evisa.mfa.mn/"],
    "uae":             ["https://u.ae/en/information-and-services/visa-and-emirates-id"],
    "united arab emirates": ["https://u.ae/en/information-and-services/visa-and-emirates-id"],
    "turkey":          ["https://www.evisa.gov.tr/en/"],
    "saudi arabia":    ["https://visa.mofa.gov.sa/"],
    "qatar":           ["https://portal.moi.gov.qa/wps/portal/MOIInternet/Visa"],
    "oman":            ["https://evisa.rop.gov.om/"],
    "kuwait":          ["https://evisa.moi.gov.kw/"],
    "bahrain":         ["https://www.evisa.gov.bh/"],
    "jordan":          ["https://www.nja.gov.jo/"],
    "egypt":           ["https://visa2egypt.gov.eg/"],
    "iran":            ["https://evisairan.ir/"],
    "ukraine":         ["https://visaonline.gov.ua/"],
    "georgia":         ["https://www.geoconsul.gov.ge/HtmlPage/html/VisaInfoEn"],
    "russia":          ["https://visa.kdmid.ru/"],
    "kazakhstan":      ["https://www.evisa.gov.kz/"],
    "uzbekistan":      ["https://e-visa.gov.uz/"],
    "brazil":          ["https://www.gov.br/mre/en/subjects/visas"],
    "argentina":       ["https://cancilleria.gob.ar/en/services/visas"],
    "mexico":          ["https://consulmex.sre.gob.mx/"],
    "colombia":        ["https://www.cancilleria.gov.co/en/procedures_services/visa"],
    "peru":            ["https://www.rree.gob.pe/servicioalciudadano/Paginas/visas.aspx"],
    "chile":           ["https://www.chileatiende.gob.cl/fichas/ver/5340"],
    "south africa":    ["https://www.dha.gov.za/index.php/applying-for-sa-visa"],
    "kenya":           ["https://evisa.go.ke/"],
    "tanzania":        ["https://visa.immigration.go.tz/"],
    "ethiopia":        ["https://www.evisa.gov.et/"],
    "nigeria":         ["https://portal.immigration.gov.ng/"],
    "ghana":           ["https://www.mfa.gov.gh/visa-application/"],
    "morocco":         ["https://www.diplomatie.ma/en/visa"],
    "tunisia":         ["https://www.diplomatie.gov.tn/en/visas/"],
}

# Capital cities used to pin queries to the actual consulate location
_CAPITALS: dict[str, str] = {
    "afghanistan": "kabul", "albania": "tirana", "algeria": "algiers",
    "argentina": "buenos aires", "armenia": "yerevan", "australia": "canberra",
    "austria": "vienna", "azerbaijan": "baku", "bangladesh": "dhaka",
    "belgium": "brussels", "bolivia": "la paz", "brazil": "brasilia",
    "bulgaria": "sofia", "cambodia": "phnom penh", "canada": "ottawa",
    "chile": "santiago", "china": "beijing", "colombia": "bogota",
    "croatia": "zagreb", "czech republic": "prague", "denmark": "copenhagen",
    "ecuador": "quito", "egypt": "cairo", "ethiopia": "addis ababa",
    "finland": "helsinki", "france": "paris", "georgia": "tbilisi",
    "germany": "berlin", "ghana": "accra", "greece": "athens",
    "hungary": "budapest", "india": "new delhi", "indonesia": "jakarta",
    "iran": "tehran", "iraq": "baghdad", "ireland": "dublin",
    "israel": "jerusalem", "italy": "rome", "ivory coast": "yamoussoukro",
    "japan": "tokyo", "jordan": "amman", "kazakhstan": "astana",
    "kenya": "nairobi", "south korea": "seoul", "kuwait": "kuwait city",
    "kyrgyzstan": "bishkek", "laos": "vientiane", "lebanon": "beirut",
    "libya": "tripoli", "malaysia": "kuala lumpur", "mexico": "mexico city",
    "moldova": "chisinau", "mongolia": "ulaanbaatar", "morocco": "rabat",
    "mozambique": "maputo", "myanmar": "naypyidaw", "nepal": "kathmandu",
    "netherlands": "amsterdam", "new zealand": "wellington",
    "nigeria": "abuja", "norway": "oslo", "oman": "muscat",
    "pakistan": "islamabad", "peru": "lima", "philippines": "manila",
    "poland": "warsaw", "portugal": "lisbon", "qatar": "doha",
    "romania": "bucharest", "russia": "moscow", "saudi arabia": "riyadh",
    "senegal": "dakar", "serbia": "belgrade", "singapore": "singapore",
    "slovakia": "bratislava", "south africa": "pretoria", "spain": "madrid",
    "sri lanka": "colombo", "sweden": "stockholm", "switzerland": "bern",
    "syria": "damascus", "taiwan": "taipei", "tajikistan": "dushanbe",
    "tanzania": "dodoma", "thailand": "bangkok", "tunisia": "tunis",
    "turkey": "ankara", "turkmenistan": "ashgabat", "uae": "abu dhabi",
    "united arab emirates": "abu dhabi", "uk": "london",
    "united kingdom": "london", "ukraine": "kyiv", "uruguay": "montevideo",
    "usa": "washington dc", "united states": "washington dc",
    "uzbekistan": "tashkent", "venezuela": "caracas", "vietnam": "hanoi",
    "yemen": "sanaa", "zambia": "lusaka", "zimbabwe": "harare",
}

# Key visa-related search terms in the destination country's local language.
# These surface official embassy/consulate pages that may never appear in
# English-only searches.
_DEST_LOCAL_TERMS: dict[str, str] = {
    "argentina":       "visa consulado embajada argentina",
    "austria":         "visum botschaft konsulat österreich",
    "belgium":         "visa ambassade consulat belgique",
    "brazil":          "visto consulado embaixada brasil itamaraty",
    "chile":           "visa consulado embajada chile",
    "china":           "签证 领事馆 大使馆 中国",
    "colombia":        "visa consulado embajada colombia",
    "czech republic":  "vízum konzulát velvyslanectví",
    "denmark":         "visum konsulat ambassade",
    "ecuador":         "visa consulado embajada ecuador",
    "egypt":           "تأشيرة قنصلية سفارة مصر",
    "finland":         "viisumi konsulaatti suurlähetystö",
    "france":          "visa consulat ambassade france",
    "germany":         "visum botschaft konsulat deutschland",
    "greece":          "βίζα προξενείο πρεσβεία ελλάδα",
    "hungary":         "vízum konzulátus nagykövetség magyarország",
    "indonesia":       "visa konsulat kedutaan indonesia",
    "iran":            "ویزا کنسولگری سفارت ایران",
    "italy":           "visto consolato ambasciata italia",
    "japan":           "ビザ 領事館 大使館 日本 査証",
    "jordan":          "تأشيرة قنصلية سفارة الأردن",
    "malaysia":        "visa konsulat kedutaan malaysia",
    "mexico":          "visa consulado embajada mexico",
    "morocco":         "visa consulat ambassade maroc",
    "netherlands":     "visum consulaat ambassade nederland",
    "norway":          "visum konsulat ambassade norge",
    "peru":            "visa consulado embajada peru",
    "philippines":     "visa konsulat embahada pilipinas",
    "poland":          "wiza konsulat ambasada polska",
    "portugal":        "visto consulado embaixada portugal",
    "russia":          "виза консульство посольство россия",
    "saudi arabia":    "تأشيرة قنصلية سفارة السعودية",
    "south korea":     "비자 영사관 대사관 한국",
    "spain":           "visado consulado embajada españa",
    "sweden":          "visum konsulat ambassad sverige",
    "switzerland":     "visum konsulat botschaft schweiz",
    "thailand":        "วีซ่า สถานกงสุล สถานทูต ไทย",
    "turkey":          "vize konsolosluk büyükelçilik türkiye",
    "ukraine":         "віза консульство посольство україна",
    "uae":             "تأشيرة قنصلية سفارة الإمارات",
    "united arab emirates": "تأشيرة قنصلية سفارة الإمارات",
    "vietnam":         "visa lãnh sự quán đại sứ quán việt nam",
}


def search_topic(
    topic: str,
    nationality: str,
    destination: str,
    residence: str = "",
    city: str = "",
    max_results: int | None = None,
) -> list[dict]:
    """Search for a specific topic within the context of a visa application."""
    max_results = max_results or config.MAX_SEARCH_RESULTS
    res = residence.lower()
    dest = destination.lower()
    location = city.lower() if city else _CAPITALS.get(res, res)

    queries = [
        f"{topic} {destination} visa {city or residence}".strip(),
        f"{destination} consulate {location} {topic}".strip(),
        f"{topic} {nationality} {destination} visa official",
    ]

    local = _DEST_LOCAL_TERMS.get(dest, "")
    if local and location:
        queries.append(f"{local} {location}")

    return _run_search(queries, max_results, search_depth="basic")


def search_visa_info(
    nationality: str,
    destination: str,
    residence: str = "",
    city: str = "",
    purpose: str = "tourism",
    max_results: int | None = None,
) -> list[dict]:
    """Return a ranked list of search results about visa requirements.

    Official government/embassy URLs are sorted to the top.
    Curated OFFICIAL_SOURCES for the destination are always prepended.
    Each result: {url, title, snippet, content, query, official}
    """
    max_results = max_results or config.MAX_SEARCH_RESULTS
    queries = _build_queries(nationality, destination, residence, city, purpose)
    search_hits = _run_search(queries, max_results, search_depth="advanced")

    # Prepend curated official sources so they are always scraped first
    official_hits = _official_source_hits(destination)
    seen = {h["url"] for h in official_hits}
    deduped = [h for h in search_hits if h["url"] not in seen]
    return official_hits + deduped


# ── Helpers ───────────────────────────────────────────────────────────────────

def _official_source_hits(destination: str) -> list[dict]:
    """Return pre-defined official source stubs for a destination."""
    urls = OFFICIAL_SOURCES.get(destination.lower(), [])
    return [
        {"url": url, "title": "", "snippet": "", "content": "", "official": True, "query": "official"}
        for url in urls
    ]


def _run_search(queries: list[str], max_results: int, search_depth: str = "advanced") -> list[dict]:
    """Try Tavily first; fall back to DuckDuckGo."""
    if config.TAVILY_API_KEY:
        results = _run_tavily_search(queries, max_results, search_depth)
        if results:
            return results
    return _run_ddgs_search(queries, max_results)


def _run_tavily_search(queries: list[str], max_results: int, search_depth: str) -> list[dict]:
    """Search via Tavily API; returns [] on any failure."""
    try:
        from tavily import TavilyClient
    except ImportError:
        return []

    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    results: list[dict] = []
    seen_urls: set[str] = set()

    for query in queries:
        if len(results) >= max_results * 2:
            break
        try:
            response = client.search(
                query=query,
                search_depth=search_depth,
                max_results=5,
                include_raw_content=True,
            )
            for r in response.get("results", []):
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append({
                        "url": url,
                        "title": r.get("title", ""),
                        "snippet": r.get("content", ""),
                        # raw_content is the full cleaned page text — skip scraping
                        "content": r.get("raw_content") or "",
                        "query": query,
                        "official": _is_official(url),
                    })
        except Exception:
            continue

    if not results:
        return []

    results.sort(key=lambda r: r["official"], reverse=True)
    return results[:max_results]


def _run_ddgs_search(queries: list[str], max_results: int) -> list[dict]:
    """Search via DuckDuckGo (free, no key required)."""
    results: list[dict] = []
    seen_urls: set[str] = set()
    errors: list[str] = []

    with DDGS() as ddgs:
        for query in queries:
            if len(results) >= max_results * 2:
                break
            try:
                for hit in ddgs.text(query, max_results=5, safesearch="off", region="wt-wt"):
                    url = hit.get("href", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append(
                            {
                                "url": url,
                                "title": hit.get("title", ""),
                                "snippet": hit.get("body", ""),
                                "content": "",  # DuckDuckGo never returns full content
                                "query": query,
                                "official": _is_official(url),
                            }
                        )
            except Exception as exc:
                errors.append(str(exc))
                continue

    if errors and not results:
        raise RuntimeError(f"All search queries failed. First error: {errors[0]}")

    results.sort(key=lambda r: r["official"], reverse=True)
    return results[:max_results]


def _build_queries(
    nationality: str, destination: str, residence: str, city: str, purpose: str
) -> list[str]:
    nat = nationality.lower()
    dest = destination.lower()
    res = residence.lower()
    # City beats capital: if the user told us they live in Geneva, search Geneva
    location = city.lower() if city else _CAPITALS.get(res, "")

    queries = []

    if location:
        # Primary: target the exact city — this is where the consulate is
        queries += [
            f"{dest} consulate {location} visa {nat}",
            f"{dest} embassy {location} {nat} visa requirements",
            f"apply {dest} visa {location} {nat} official",
        ]

    if res and res != location:
        # Broad country-level fallback queries
        queries += [
            f"{dest} consulate {res} visa {nat} official",
            f"{dest} embassy {res} visa application {nat}",
        ]

    queries += [
        f"visa requirements {nat} citizens {dest} official government",
        f"{dest} visa {nat} entry requirements {purpose}",
        f"{dest} immigration {nat} official",
    ]

    # Local-language query: official consulate pages are often only in the
    # destination country's language and won't appear in English searches.
    local = _DEST_LOCAL_TERMS.get(dest, "")
    if local and location:
        queries.append(f"{local} {location}")
    elif local and res:
        queries.append(f"{local} {res}")

    return queries


def _is_official(url: str) -> bool:
    lower = url.lower()
    return any(token in lower for token in _OFFICIAL_DOMAINS)
