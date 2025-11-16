
def get_all_statuses(data):
    statuses = set()
    for battle in data:
        for turn in battle.get("battle_timeline", []):
            for side in ["p1_pokemon_state", "p2_pokemon_state"]:
                state = turn.get(side, {})
                status = state.get("status")
                if status is not None:
                    statuses.add(status)
    return statuses

def get_pokemon_stats(data):
    pokemon_stats = {}

    for battle in data:
        for p in battle["p1_team_details"]:
            name = p["name"].lower()
            if name not in pokemon_stats:
                pokemon_stats[name] = {
                    "base_hp": p.get("base_hp",0),
                    "base_atk": p.get("base_atk", 0),
                    "base_def": p.get("base_def", 0),
                    "base_spa": p.get("base_spa", 0),
                    "base_spd": p.get("base_spd", 0),
                    "base_spe": p.get("base_spe",0)
                }
    return pokemon_stats


attack_types = {
    "normal": {"rock": 0.5, "ghost": 0},
    "fire": {"grass": 2, "fire": 0.5, "water": 0.5, "bug": 2, "rock": 0.5, "ice": 2, "dragon": 0.5},
    "water": {"fire": 2, "water": 0.5, "grass": 0.5, "ground": 2, "rock": 2, "dragon": 0.5},
    "electric": {"water": 2, "electric": 0.5, "grass": 0.5, "ground": 0, "flying": 2, "dragon": 0.5},
    "grass": {"water": 2, "fire": 0.5, "grass": 0.5, "poison": 0.5, "ground": 2, "flying": 0.5, "bug": 0.5, "rock": 2, "dragon": 0.5},
    "ice": {"grass": 2, "ice": 0.5, "water": 0.5, "ground": 2, "flying": 2, "dragon": 2},
    "fighting": {"normal": 2, "rock": 2, "ice": 2, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "ghost": 0},
    "poison": {"grass": 2, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5},
    "ground": {"fire": 2, "electric": 2, "grass": 0.5, "poison": 2, "flying": 0, "bug": 0.5, "rock": 2},
    "flying": {"grass": 2, "electric": 0.5, "fighting": 2, "bug": 2, "rock": 0.5},
    "psychic": {"fighting": 2, "poison": 2, "psychic": 0.5},
    "bug": {"grass": 2, "fire": 0.5, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2, "ghost": 0.5},
    "rock": {"fire": 2, "ice": 2, "fighting": 0.5, "ground": 0.5, "flying": 2, "bug": 2},
    "ghost": {"ghost": 2, "psychic": 0},
    "dragon": {"dragon": 2}
}

types = {
    "bulbasaur": ["grass", "poison"], "ivysaur": ["grass", "poison"], "venusaur": ["grass", "poison"],
    "charmander": "fire", "charmeleon": "fire", "charizard": ["fire", "flying"],
    "squirtle": "water", "wartortle": "water", "blastoise": "water",
    "caterpie": "bug", "metapod": "bug", "butterfree": ["bug", "flying"],
    "weedle": ["poison", "bug"], "kakuna": ["poison", "bug"], "beedrill": ["poison", "bug"],
    "pidgey": ["normal", "flying"], "pidgeotto": ["normal", "flying"], "pidgeot": ["normal", "flying"],
    "rattata": "normal", "raticate": "normal", "spearow": ["normal", "flying"],
    "fearow": ["normal", "flying"], "ekans": "poison", "arbok": "poison",
    "pikachu": "electric", "raichu": "electric", "sandshrew": "ground", "sandslash": "ground",
    "nidoran♀": "poison", "nidorina": "poison", "nidoqueen": ["poison", "ground"],
    "nidoran♂": "poison", "nidorino": "poison", "nidoking": ["poison", "ground"],
    "clefairy": "normal", "clefable": "normal", "vulpix": "fire", "ninetales": "fire",
    "zubat": ["poison", "flying"], "golbat": ["poison", "flying"], "oddish": ["grass", "poison"],
    "gloom": ["grass", "poison"], "vileplume": ["grass", "poison"], "paras": ["bug", "grass"],
    "parasect": ["bug", "grass"], "venonat": ["bug", "poison"], "venomoth": ["bug", "poison"],
    "diglett": "ground", "dugtrio": "ground", "meowth": "normal", "persian": "normal",
    "psyduck": "water", "golduck": "water","mankey":"fighting","primeape":"fighting","growlithe":"fire",
    "arcanine":"fire","poliwag":"water","poliwhirl":"water","poliwrath":["water","fighting"],"abra":"psychic",
    "kadabra":"psychic","alakazam":"psychic","machop": "fighting", "machoke": "fighting",
    "machamp": "fighting", "bellsprout": ["grass", "poison"], "weepinbell": ["grass", "poison"],
    "victreebel": ["grass", "poison"], "tentacool": ["water", "poison"], "tentacruel": ["water", "poison"],
    "geodude": ["rock", "ground"], "graveler": ["rock", "ground"], "golem": ["rock", "ground"],
    "ponyta": "fire", "rapidash": "fire", "slowpoke": ["water", "psychic"], "slowbro": ["water", "psychic"],
    "magnemite": "electric", "magneton": "electric", "farfetch'd": ["normal", "flying"],
    "doduo": ["normal", "flying"], "dodrio": ["normal", "flying"], "seel": "water", "dewgong": ["water", "ice"],
    "grimer": "poison", "muk": "poison", "shellder": "water", "cloyster": ["water", "ice"],
    "gastly": ["ghost", "poison"], "haunter": ["ghost", "poison"], "gengar": ["ghost", "poison"],
    "onix": ["rock", "ground"], "drowzee": "psychic", "hypno": "psychic", "krabby": "water", "kingler": "water",
    "exeggcute": ["grass", "psychic"], "exeggutor": ["grass", "psychic"], "cubone": "ground", "marowak": "ground",
    "lickitung": "normal", "koffing": "poison", "weezing": "poison", "rhyhorn": ["rock", "ground"],
    "rhydon": ["rock", "ground"], "chansey": "normal", "tangela": "grass", "kangaskhan": "normal",
    "horsea": "water", "seadra": "water", "goldeen": "water", "seaking": "water",
    "staryu": "water", "starmie": ["water", "psychic"], "mr. mime": "psychic", "scyther": ["bug", "flying"],
    "jynx": ["ice", "psychic"], "electabuzz": "electric", "magmar": "fire", "pinsir": "bug",
    "tauros": "normal", "magikarp": "water", "gyarados": ["water", "flying"], "lapras": ["water", "ice"],
    "ditto": "normal", "eevee": "normal", "vaporeon": "water", "jolteon": "electric", "flareon": "fire",
    "porygon": "normal", "omanyte": ["rock", "water"], "omastar": ["rock", "water"], "kabuto": ["rock", "water"],
    "kabutops": ["rock", "water"], "aerodactyl": ["rock", "flying"], "mew": "psychic", "mewtwo": "psychic",
    "voltorb": "electric",
    "electrode": "electric",
    "hitmonlee": "fighting",
    "hitmonchan": "fighting",
    "articuno": ["ice", "flying"],
    "zapdos": ["electric", "flying"],
    "moltres": ["fire", "flying"],
    "dratini": "dragon",
    "dragonair": "dragon",
    "dragonite": ["dragon", "flying"],
    "snorlax": "normal"
}

def reconstruct_p2_team(battle, pokemon_stats, types):
    p2_team = {}
    
    for turn in battle["battle_timeline"]:
        p2_state = turn.get("p2_pokemon_state", {})
        if not p2_state or "name" not in p2_state:
            continue
        
        name = p2_state["name"].lower().strip()
        if name not in p2_team:
            p2_team[name] = {
                "base_hp": 0,
                "base_atk": 0,
                "base_def": 0,
                "base_spa": 0,
                "base_spd": 0,
                "base_spe": 0,
                "types": types.get(name, [])
            }

        if name in pokemon_stats:
            for stat in ["base_hp", "base_atk", "base_def", "base_spa", "base_spd","base_spe"]:
                if p2_team[name][stat] == 0:
                    p2_team[name][stat] = pokemon_stats[name].get(stat, 0)
    
    return p2_team

def extract_unique_moves(data):
    moves = set()  
    
    for battle in data:
        for turn in battle.get("battle_timeline", []):
            p1_move = turn.get("p1_move_details")
            if p1_move and "name" in p1_move:
                moves.add(p1_move["name"].lower().strip())
                p2_move = turn.get("p2_move_details")
            if p2_move and "name" in p2_move:
                moves.add(p2_move["name"].lower().strip())
    return moves