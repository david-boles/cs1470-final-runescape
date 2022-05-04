import json

def get_observation(timestamp):
    endpoint = "https://prices.runescape.wiki/api/v1/osrs/1h"
    headers = {"User-Agent": "bit_n #0065; predictive modelling"}
    
    r = requests.get(f"{endpoint}?timestamp={timestamp}",headers=headers)
    return r.json()['data']
