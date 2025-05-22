import re
import pandas as pd

# Update the path if needed
DATA_PATH = "data/influencers.csv"

DOMAIN_KEYWORDS = {
    "fashion": ["fashion", "clothes", "outfit", "style"],
    "beauty": ["makeup", "beauty", "skincare", "cosmetics"],
    "tech": ["technology", "gadgets", "devices", "smartphone"],
    "fitness": ["fitness", "gym", "workout", "exercise"],
    "food": ["food", "cooking", "recipe", "restaurant"],
    "travel": ["travel", "vacation", "trip", "tourism"],
    "gaming": ["gaming", "video games", "esports"],
    "finance": ["finance", "investment", "crypto", "money"],
}

COUNTRY_KEYWORDS = {
    "ma": ["ma", "morocco", "moroccan"],
    "us": ["us", "usa", "united states", "america", "american"],
    "uk": ["uk", "united kingdom", "england", "british"],
    "fr": ["fr", "france", "french"],
    "de": ["de", "germany", "german"],
    "in": ["in", "india", "indian"],
    "jp": ["jp", "japan", "japanese"],
    "ca": ["ca", "canada", "canadian"],
    "tn": ["tn", "Tunisia", "tunisian"],
    "tr": ["tr", "turkey", "turkish"],
    "dz": ["dz", "algeria", "algerian"],
    "it": ["it", "italy", "italian"],
    "es": ["es", "spain", "spanish"],
    "kr": ["kr", "south korea", "korean", "south-korea"],
    "br": ["br", "brazil", "brazilian"],
    "ar": ["ar", "argentina", "argentinian"],
    "au": ["au", "australia", "australian"],
    "ch": ["ch", "switzerland", "swiss"],
    "be": ["be", "belgium", "belgian"],
    "ae": ["ae", "united arab emirates", "emirati", "uae"]
}


def detect_domain_and_country(prompt: str):
    prompt = prompt.lower()

    # Detect domain
    domain_detected = None
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(re.search(rf'\b{re.escape(kw)}\b', prompt) for kw in keywords):
            domain_detected = domain
            break
    if domain_detected is None:
        domain_detected = "beauty"

    # Detect country - first check for exact matches in the prompt
    country_detected = None
    for code, keywords in COUNTRY_KEYWORDS.items():
        if any(re.search(rf'\b{re.escape(kw)}\b', prompt) for kw in keywords):
            country_detected = code
            break
    
    # Special case for Tunisia (make sure it's matching "tunisia" in the prompt)
    if "tunisia" in prompt or "tunisian" in prompt:
        country_detected = "tn"
        
    if country_detected is None:
        country_detected = "ma"  # Default to Morocco

    return domain_detected, country_detected


def get_influencers(domain: str, country: str, budget: float = None):
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return pd.DataFrame()

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Convert country code to match CSV (simplified version)
    country = country.lower()
    if country == "tn":
        country = "tunisia"
    elif country == "ma":
        country = "morocco"
    # Add other country mappings as needed based on your CSV data

    # Make country matching more flexible
    country_pattern = re.compile(rf'\b{re.escape(country)}\b', flags=re.IGNORECASE)
    
    # Filter by domain and country
    filtered = df[
        (df["topics"].str.lower().str.contains(domain)) &
        (df["country"].str.lower().str.contains(country_pattern))
    ].copy()

    # If no results, try relaxing the filters
    if filtered.empty:
        filtered = df[
            (df["topics"].str.lower().str.contains(domain)) |
            (df["country"].str.lower().str.contains(country_pattern))
        ].copy()

    # Parse followers
    def parse_followers(value):
        try:
            if isinstance(value, str):
                value = value.lower().replace(",", "").replace("k", "000").replace("m", "000000")
                return float(''.join(c for c in value if c.isdigit() or c == '.'))
            return float(value)
        except:
            return 0

    if not filtered.empty:
        filtered.loc[:, "followers_num"] = filtered["followers"].apply(parse_followers)
        filtered = filtered.sort_values(by="followers_num", ascending=False)

    return filtered.head(5)

    