from candidate_vetting.public_catalogs import static_catalogs
import requests
from typing import List
import os
import numpy as np

def get_bibtex_from_bibcodes(bibcodes, api_key=None):
    """Fetch BibTeX citations from NASA ADS API."""

    bibcodes = clean_bibcode_list(bibcodes)
    if not bibcodes:
        return ""
    
    # Debug: Check if API key exists
    if api_key is None:
        api_key = os.environ.get("ADS_DEV_KEY")
        print(f"[DEBUG] API key from environment: {api_key[:20] if api_key else 'NOT FOUND'}...")
    else:
        print(f"[DEBUG] API key provided: {api_key[:20]}...")
    
    if not api_key:
        raise ValueError("No API key found. Set ADS_DEV_KEY environment variable or pass api_key parameter.")
    
    url = "https://api.adsabs.harvard.edu/v1/export/bibtex"
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'bibcode': bibcodes  # Pass as a list
    }
    
    #print(f"[DEBUG] Payload: {payload}")
    #print(f"[DEBUG] Headers (without key): {{'Authorization': 'Bearer ***', 'Content-Type': 'application/json'}}")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        #print(f"[DEBUG] Status code: {response.status_code}")
        #print(f"[DEBUG] Response: {response.text}")
        response.raise_for_status()
        result = response.json()
        return result.get('export', '')
    
    except requests.RequestException as e:
        print(f"Error fetching BibTeX: {e}")
        #raise
        return ""

def clean_bibcode_list(bibcodes):
    return list(np.unique([x for x in bibcodes if len(x)]))
    
bibcode_map = {}
all_bibcodes = []
for label, cl in static_catalogs.__dict__.items():
    try:
        cat = cl()
        print(label, cat.ads_urls, cat.catalog_version)
        bibcodes = cat.bibcodes
    except:
        continue
    print(bibcodes)

    try:
        bibcode_map[str(cat)] = bibcodes
    except TypeError:
        pass
    all_bibcodes += bibcodes
    
bibtex = get_bibtex_from_bibcodes(all_bibcodes, api_key=os.environ.get("ADS_DEV_KEY"))
    
print(bibtex)
for key, bibcodes in bibcode_map.items():
    print(key, f'\\citep{{{",".join(bibcodes)}}}')
