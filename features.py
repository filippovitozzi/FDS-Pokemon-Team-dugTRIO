import numpy as np
import pandas as pd
from helpers import reconstruct_p2_team


def build_features(data, pokemon_stats, attack_types,types,is_train=True):
    rows=[]
    for battle in data:
        row={
            "round_count_ratio":round_count_ratio(battle),
            "speed_diff":speed_diff(battle, pokemon_stats, types),
            "physical_vs_special_diff":physical_vs_special_diff (battle,pokemon_stats,types),
            "battle_id" : battle["battle_id"],
            "last_turn_statues":last_turn_statues(battle),
            "effective_status_diff":status_diff(battle),
            "coverage_diff":coverage_diff(battle, types),
            "net_attack_advantage": net_attack_advantage(battle,attack_types,types),
            "stab_diff":stab_diff(battle,types),
            "alive_diff":alive_diff(battle, types),
            "smart_switch_difference":smart_switch_difference(battle,attack_types,types),
            "difference_hp_ratio": difference_hp_ratio(battle,pokemon_stats,types),
            "total_stats_diff":total_stats_diff(battle, pokemon_stats, types),
            "weak_attack_diff":weak_attack_diff(battle,attack_types,types, neutral=0.5, smoothing=0),
            "strong_attack_difference":strong_attack_difference(battle, attack_types, types),
            "first_strike_feature_boostaware":first_strike_feature_boostaware(battle, pokemon_stats, types),
            "hp_momentum":hp_momentum(battle),
            "lead_difference":lead_difference(battle, attack_types),
            "variance_difference":variance_difference(battle,pokemon_stats,types),
            "attack_diff":attack_diff(battle, pokemon_stats, types),
            "defense_diff":defense_diff(battle, pokemon_stats, types),
            "special_attack_diff":special_attack_diff(battle, pokemon_stats, types),
            "special_defense_diff":special_defense_diff(battle, pokemon_stats, types),
            "hp_diff":hp_diff(battle, pokemon_stats, types)   
        }
        
        if is_train and "player_won" in battle:
            row["player_won"] = int(battle["player_won"])
        rows.append(row)
    features_df = pd.DataFrame(rows)
    return features_df



def round_count_ratio(battle):
    count_p1=0
    for turn in battle["battle_timeline"]:
        if turn["p1_move_details"] is not None:
            count_p1 += 1
    count_p2=0
    for turn in battle["battle_timeline"]:
        if turn["p2_move_details"] is not None:
            count_p2 += 1
    if count_p2!=0:
        if count_p2==0:
            return 0.0
        return float(count_p1/count_p2)
    else:
        return 1

def last_turn_statues(battle):
    list_of_bad_statues=["brn","tox","par","psn","frz"]
    last_round=battle["battle_timeline"][-1]
    if last_round["p1_pokemon_state"]["status"] in list_of_bad_statues:
        return 1
    else:
        return 0

def physical_vs_special_p1(battle):
    role_map = {}
    for pokemon in battle["p1_team_details"]:
        if pokemon["base_atk"] > pokemon["base_spa"]:
            role_map[pokemon["name"]] = "PHYSICAL"
        else:
            role_map[pokemon["name"]] = "SPECIAL"
    count_right=0
    correct=True
    count_p1=0
    for turn in battle["battle_timeline"]:
        if turn["p1_move_details"] is None:
            count_p1 += 1
        if turn["p1_move_details"] is None:
            continue
        pokemon_name=turn["p1_pokemon_state"]["name"]
        move_type=turn["p1_move_details"]["category"]
        if role_map[pokemon_name]==move_type:
            count_right+=1
            pass
        else:
            correct=False
    if correct:
        return 1
    else:
        if (30-count_p1)==0:
            return 0.0
        return float(count_right/(30-count_p1))  

def physical_vs_special_p2(battle, pokemon_stats, types):
    p2_team = reconstruct_p2_team(battle, pokemon_stats, types)  
    role_map = {}
    for name_lower, stats in p2_team.items():
        if stats.get("base_atk", 0) > stats.get("base_spa", 0):
            role_map[name_lower] = "PHYSICAL"
        else:
            role_map[name_lower] = "SPECIAL"

    count_right = 0
    correct = True
    count_p2 = 0

    for turn in battle["battle_timeline"]:
        if turn["p2_move_details"] is None:
            count_p2 += 1
            continue

        pokemon_name = (turn["p2_pokemon_state"]["name"]).lower().strip()
        move_type = turn["p2_move_details"]["category"]  

        if pokemon_name not in role_map:
            continue

        if role_map[pokemon_name] == move_type:
            count_right += 1
        else:
            correct = False

    if correct:
        return 1
    else:
        denom = 30 - count_p2  
        if denom == 0:
            return 0.0
        return float(count_right / denom)

def physical_vs_special_diff (battle,pokemon_stats,types):
    p1=physical_vs_special_p1(battle)
    p2=physical_vs_special_p2(battle,pokemon_stats,types)
    return p1 - p2

def _norm_name(n):
    return (n or "").lower().strip()

def _stage_mult(stage):
    try:
        k = int(stage)
    except:
        k = 0
    return (2 + k) / 2 if k >= 0 else 2 / (2 - k)

def _effective_speed(base_spe, boosts_dict=None, include_boosts=False):
    spe = float(base_spe or 0)
    if include_boosts and boosts_dict:
        mult = _stage_mult(boosts_dict.get("spe", 0))
        spe *= mult
    return spe

def first_strike_stats(battle, pokemon_stats, types, include_boosts=False):
    p1_speed = {}
    for p in (battle.get("p1_team_details") or []):
        p1_speed[_norm_name(p.get("name"))] = int(p.get("base_spe", 0))

    p2_team = reconstruct_p2_team(battle, pokemon_stats, types)
    p2_speed = {name: int(stats.get("base_spe", 0)) for name, stats in p2_team.items()}

    p1_first = 0
    p2_first = 0
    both_attack_turns = 0
    ties = 0

    for t in (battle.get("battle_timeline") or [])[:30]:
        md1 = t.get("p1_move_details")
        md2 = t.get("p2_move_details")

        if md1 is None or md2 is None:
            continue

        both_attack_turns += 1

        p1_name = _norm_name((t.get("p1_pokemon_state") or {}).get("name"))
        p2_name = _norm_name((t.get("p2_pokemon_state") or {}).get("name"))
        p1_boosts = (t.get("p1_pokemon_state") or {}).get("boosts") or {}
        p2_boosts = (t.get("p2_pokemon_state") or {}).get("boosts") or {}

        sp1 = _effective_speed(p1_speed.get(p1_name, 0), p1_boosts, include_boosts)
        sp2 = _effective_speed(p2_speed.get(p2_name, 0), p2_boosts, include_boosts)

        if sp1 > sp2:
            p1_first += 1
        elif sp2 > sp1:
            p2_first += 1
        else:
            ties += 1
            
    if both_attack_turns - ties > 0:
        diff_norm = (p1_first - p2_first) / float(both_attack_turns - ties)
    else:
        diff_norm = 0.0

    return {
        "p1_first_count": int(p1_first),
        "p2_first_count": int(p2_first),
        "both_attack_turns": int(both_attack_turns),
        "tie_speed_turns": int(ties),
        "first_strike_diff_norm": float(diff_norm)
     }

def first_strike_feature(battle, pokemon_stats, types, include_boosts=False):
    out = first_strike_stats(battle, pokemon_stats, types, include_boosts)
    return out["first_strike_diff_norm"]

def total_stats_diff(battle, pokemon_stats, types):
    STAT_KEYS = ["base_hp","base_atk","base_def","base_spa","base_spd","base_spe"]

    p1_team = battle.get("p1_team_details") or []

    p2_team_dict = reconstruct_p2_team(battle, pokemon_stats, types)
    p2_team = list(p2_team_dict.values())

    team_1_tot = 0
    team_2_tot = 0

    for mon in p1_team:
        for stat in STAT_KEYS:
            team_1_tot += int(mon.get(stat, 0))
            
    for mon in p2_team:
        for stat in STAT_KEYS:
            team_2_tot += int(mon.get(stat, 0))

    team_1_tot_mean = team_1_tot / 6.0 if p1_team else 0.0
    team_2_den = len(p2_team) if p2_team else 1
    team_2_tot_mean = team_2_tot / float(team_2_den)

    return float(team_1_tot_mean - team_2_tot_mean)

def battle_has_speed_boosts(battle):
    for t in (battle.get("battle_timeline") or [])[:30]:
        p1s = t.get("p1_pokemon_state") or {}
        p2s = t.get("p2_pokemon_state") or {}
        b1 = (p1s.get("boosts") or {}).get("spe", 0)
        b2 = (p2s.get("boosts") or {}).get("spe", 0)
        if b1 or b2:
            return True
    return False

def first_strike_stats_boostaware(battle, pokemon_stats, types):
    use_boosts = battle_has_speed_boosts(battle)
    out = first_strike_stats(battle, pokemon_stats, types, include_boosts=use_boosts)
    out["used_speed_boosts"] = bool(use_boosts)
    return out

def first_strike_feature_boostaware(battle, pokemon_stats, types):
    return first_strike_stats_boostaware(battle, pokemon_stats, types)["first_strike_diff_norm"]

def p1_percentage_turn_status(battle):       
    list_of_bad_statues=["brn","tox","par","psn","frz"]
    p1_under_status=0
    for turn in battle["battle_timeline"]:
        if turn["p1_pokemon_state"]["status"] in list_of_bad_statues:
            p1_under_status+=1
    return p1_under_status/30

def p2_percentage_turn_status(battle):        
    list_of_bad_statues=["brn","tox","par","psn","frz"]
    p2_under_status=0
    for turn in battle["battle_timeline"]:
        if turn["p2_pokemon_state"]["status"] in list_of_bad_statues:
            p2_under_status+=1
    return p2_under_status/30

def status_diff(battle):
    p1 = p1_percentage_turn_status(battle)
    p2 = p2_percentage_turn_status(battle)
    return p2 - p1

def p2_team(battle,types):
    p2_team={}
    for turn in battle["battle_timeline"]:
        current_name=turn["p2_pokemon_state"]["name"].lower()
        if current_name in types:
            p2_team[current_name]=types[current_name]
        else:
            pass
    return p2_team

def p1_coverage(battle):
    possible_types= {
    "normal": 0,
    "fire": 0,
    "fighting": 0,
    "water": 0,
    "flying": 0,
    "grass": 0,
    "poison": 0,
    "electric": 0,
    "ground": 0,
    "psychic": 0,
    "rock": 0,
    "ice": 0,
    "bug": 0,
    "dragon": 0,
    "ghost": 0
}
    for pokemon in battle["p1_team_details"]:
        for element in pokemon["types"]:
            if element in possible_types:
                possible_types[element]+=1
    count_types=0
    for key in possible_types:
        if possible_types[key]>0:
            count_types+=1
    return count_types/len(possible_types.keys())

def p2_coverage(battle,types):
    p2=p2_team(battle,types)
    p2_types=set()
    for value in p2.values():
        if isinstance(value,str):
            p2_types.add(value)
        if isinstance(value,list):
            for i in range(len(value)):
                 p2_types.add(value[i])
    return len(p2_types)/15

def coverage_diff(battle, types):
    return p1_coverage(battle) - p2_coverage(battle, types)

def strong_attack_p1(battle, attack_types, types):
    p2 = p2_team(battle, types)  

    count_strong_attack = 0
    count_p1 = 0

    for turn in battle["battle_timeline"]:
        move = turn.get("p1_move_details")
        if not move:
            continue

        move_type_p1 = move.get("type", "").lower()
        if move_type_p1 not in attack_types:
            continue

        count_p1 += 1  

        p2_state = turn.get("p2_pokemon_state", {})
        p2_name = p2_state.get("name", "").lower()
        p2_pokemon_type = p2.get(p2_name)
        if not p2_pokemon_type:
            continue

        types_to_check = (
            [p2_pokemon_type.lower()] if isinstance(p2_pokemon_type, str)
            else [t.lower() for t in p2_pokemon_type]
        )

        for t in types_to_check:
            if attack_types[move_type_p1].get(t, 1) == 2:
                count_strong_attack += 1
                break  

    return count_strong_attack / count_p1 if count_p1 > 0 else 0.0

def strong_attack_p2(battle, attack_types, types):
    p1 = {p["name"].lower(): [t.lower() for t in p["types"]] for p in battle["p1_team_details"]}
    
    count_strong_attack = 0
    count_p2 = 0

    for turn in battle["battle_timeline"]:
        move = turn.get("p2_move_details")
        if not move:
            continue
        
        move_type_p2 = move.get("type", "").lower()
        if move_type_p2 not in attack_types:
            continue
        
        count_p2 += 1  
        
        p1_state = turn.get("p1_pokemon_state", {})
        p1_name = p1_state.get("name", "").lower()
        p1_pokemon_type = p1.get(p1_name)
        if not p1_pokemon_type:
            continue

        for t in p1_pokemon_type:
            if attack_types[move_type_p2].get(t, 1) == 2:
                count_strong_attack += 1
                break  

    return count_strong_attack / count_p2 if count_p2 > 0 else 0.0

def strong_attack_ratio(battle, attack_types, types):
    sa_p1 = strong_attack_p1(battle, attack_types, types)
    sa_p2 = strong_attack_p2(battle, attack_types, types)
    
    total = sa_p1 + sa_p2
    
    if total == 0:
        return 0.5
    
    return sa_p1 / total

def strong_attack_difference(battle, attack_types, types):
    sa_p1 = strong_attack_p1(battle, attack_types, types)
    sa_p2 = strong_attack_p2(battle, attack_types, types)
    return sa_p1 - sa_p2

def weak_attack_p1(battle, attack_types, types):
    p2 = p2_team(battle, types)
    count_weak_attack = 0
    count_p1 = 0

    for turn in battle["battle_timeline"]:
        move = turn.get("p1_move_details")
        if not move:
            continue
        count_p1 += 1

        move_type_p1 = move.get("type", "").lower()
        if move_type_p1 not in attack_types:
            continue

        p2_name = turn["p2_pokemon_state"]["name"].lower()
        p2_pokemon_type = p2.get(p2_name)
        if not p2_pokemon_type:
            continue
        types_to_check = (
            [p2_pokemon_type.lower()] if isinstance(p2_pokemon_type, str)
            else [t.lower() for t in p2_pokemon_type]
        )

        for t in types_to_check:
            if attack_types[move_type_p1].get(t, 1)== 0.5:
                count_weak_attack += 1

    return count_weak_attack / count_p1 if count_p1 > 0 else 0.0

def weak_attack_p2(battle,attack_types,types):
    p1={}
    for pokemon in battle["p1_team_details"]:
        pokemon_name=pokemon["name"]
        pokemon_types=pokemon["types"]
        p1[pokemon_name]=pokemon_types
    count_weak_attack=0
    count_p2=0
    for turn in battle["battle_timeline"]:
        move = turn.get("p2_move_details")
        if not move:
            continue
        count_p2 += 1
        move_type_p2 = move.get("type", "").lower()
        if move_type_p2 not in attack_types:
            continue
        p1_name = turn["p1_pokemon_state"]["name"].lower()
        p1_pokemon_type = p1.get(p1_name)
        if not p1_pokemon_type:
            continue
        types_to_check = (
            [p1_pokemon_type.lower()] if isinstance(p1_pokemon_type, str)
            else [t.lower() for t in p1_pokemon_type]
        )
        for t in types_to_check:
            if attack_types[move_type_p2].get(t, 1)==0.5:
                count_weak_attack += 1
    return count_weak_attack / count_p2 if count_p2 > 0 else 0.0

def weak_attack_ratio(battle,attack_types,types,*, neutral=0.5, smoothing=0):
    p1 = weak_attack_p1(battle, attack_types, types)
    p2 = weak_attack_p2(battle, attack_types, types)

    if smoothing > 0:
        return (p1 + smoothing) / (p1 + p2 + 2 * smoothing)

    denom = p1 + p2
    if denom == 0:
        return neutral
    return p1 / denom

def weak_attack_diff(battle,attack_types,types,*, neutral=0.5, smoothing=0):
    p1 = weak_attack_p1(battle, attack_types, types)
    p2 = weak_attack_p2(battle, attack_types, types)
    return p1 - p2

def net_attack_advantage(battle,attack_types,types):
    return strong_attack_ratio(battle,attack_types,types)-weak_attack_ratio(battle,attack_types,types)

def stabs_p1(battle):
    p1={}
    count_stabs=0
    count_p1=0
    for pokemon in battle["p1_team_details"]:
        pokemon_=pokemon["name"]
        pokemon_types=pokemon["types"]
        p1[pokemon_]=pokemon_types
    for turn in battle["battle_timeline"]:
        pokemon_state=turn["p1_pokemon_state"]
        move = turn.get("p1_move_details")
        if not move:
            continue
        count_p1+=1
        move_type_p1 = move.get("type", "").lower() 
        pokemon_name=pokemon_state["name"].lower()
        if pokemon_name in p1 and move_type_p1 in [t.lower() for t in p1[pokemon_name]]:
            count_stabs += 1
    return count_stabs/count_p1 if count_p1>0 else 0.0

def stabs_p2(battle,types):
    p2=p2_team(battle, types)
    count_stabs=0
    count_p2=0
    for turn in battle["battle_timeline"]:
        pokemon_state=turn["p2_pokemon_state"]
        move = turn.get("p2_move_details")
        if not move:
            continue
        count_p2+=1
        move_type_p2 = move.get("type", "").lower() 
        pokemon_name = pokemon_state["name"].lower()
        if pokemon_name in p2 and move_type_p2 in [t.lower() for t in p2[pokemon_name]]:
            count_stabs += 1
    return count_stabs/count_p2 if count_p2>0 else 0.0

def stab_diff(battle,types):
    return ((stabs_p1(battle)-stabs_p2(battle,types)))

def pokemon_alive_p1(battle):
    team = battle.get("p1_team_details", [])
    pokemon_p1 = {p["name"]: p.get("base_hp", 0) for p in team if "name" in p}
    pokemon_alive = 0

    for turn in (battle.get("battle_timeline") or []):
        p1_state = turn.get("p1_pokemon_state", {}) or {}
        if "name" in p1_state and "hp_pct" in p1_state and p1_state["hp_pct"] is not None:
            name = p1_state["name"]
            hp_pct = p1_state["hp_pct"]
            if name in pokemon_p1:
                base_hp = next(p.get("base_hp", 0) for p in team if p.get("name") == name)
                pokemon_p1[name] = (hp_pct / 100.0) * base_hp

    for hp in pokemon_p1.values():
        if hp and hp > 0:
            pokemon_alive += 1

    team_size = len(pokemon_p1) if len(pokemon_p1) > 0 else 6 
    return pokemon_alive / team_size

def pokemon_alive_p2(battle, types):
    p2 = p2_team(battle, types) or {}
    team_size = len(p2) if len(p2) > 0 else 6  # evita /0
    pokemon_alive = team_size
    
    fainted = set()
    for turn in (battle.get("battle_timeline") or []):
        p2_state = (turn.get("p2_pokemon_state") or {})
        name = p2_state.get("name")
        status = p2_state.get("status")
        if name and status == "fnt" and name not in fainted:
            fainted.add(name)
            pokemon_alive -= 1

    if pokemon_alive < 0:
        pokemon_alive = 0

    return pokemon_alive / team_size

def alive_diff(battle, types):
    return pokemon_alive_p1(battle) - pokemon_alive_p2(battle, types)

def smart_switch_ratio_p1(battle,attack_types,types):
    smart_switches=0
    stupid_switches=0
    total_switches=0
    prev_p1=None
    for turn in battle["battle_timeline"]:
        current_p1 = turn["p1_pokemon_state"]["name"]
        current_p2 = turn["p2_pokemon_state"]["name"]
        if prev_p1 is None:
            prev_p1 = current_p1
            continue
        if current_p1 != prev_p1:
            total_switches += 1
            prev_p1 = current_p1
        p1_types = types.get(current_p1, [])
        p2_types = types.get(current_p2, [])
        if isinstance(p1_types, str): p1_types = [p1_types]
        if isinstance(p2_types, str): p2_types = [p2_types]
        advantage = False
        disadvantage = False
        for t1 in p1_types:
                if t1 not in attack_types:
                    continue
                for t2 in p2_types:
                    mult = attack_types[t1].get(t2, 1)
                    if mult >= 2:
                        advantage = True
                    elif mult <= 0.5:
                        disadvantage = True
        if advantage:
                smart_switches += 1
        elif disadvantage:
                stupid_switches += 1
    if stupid_switches == 0:
        return smart_switches
    else:
        return smart_switches / stupid_switches

def smart_switch_ratio_p2(battle, attack_types, types):
    total_switches = 0
    smart_switches = 0
    stupid_switches = 0
    prev_p2 = None

    for turn in battle["battle_timeline"]:
        current_p2 = turn["p2_pokemon_state"]["name"].lower()
        current_p1 = turn["p1_pokemon_state"]["name"].lower()
        
        if prev_p2 is None:
            prev_p2 = current_p2
            continue

        if current_p2 != prev_p2:
            total_switches += 1
            prev_p2 = current_p2
            
            p2_types = types.get(current_p2, [])
            p1_types = types.get(current_p1, [])
            if isinstance(p2_types, str): p2_types = [p2_types]
            if isinstance(p1_types, str): p1_types = [p1_types]

            advantage = False
            disadvantage = False

            for t1 in p2_types:
                if t1 not in attack_types:
                    continue
                for t2 in p1_types:
                    mult = attack_types[t1].get(t2, 1)
                    if mult >= 2:
                        advantage = True
                    elif mult <= 0.5:
                        disadvantage = True
            
            if advantage:
                smart_switches += 1
            elif disadvantage:
                stupid_switches += 1
    if stupid_switches == 0:
        return smart_switches
    else:
        return smart_switches / stupid_switches

def smart_switch_difference(battle,attack_types,types):
    return smart_switch_ratio_p1(battle,attack_types,types)-smart_switch_ratio_p2(battle, attack_types, types)




def hp_variation_ratio_p1(battle):
    pokemon_p1 = {p["name"]: p["base_hp"] for p in battle["p1_team_details"]}
    if not pokemon_p1:
        return 0.0

    avg_initial_hp = sum(pokemon_p1.values()) / len(pokemon_p1)

    last_hp_pct = {name: 1.0 for name in pokemon_p1.keys()}

    for turn in battle.get("battle_timeline", []):
        p1_state = turn.get("p1_pokemon_state", {})
        if not p1_state:
            continue
        name = p1_state.get("name")
        hp_pct = p1_state.get("hp_pct")
        if name in last_hp_pct and isinstance(hp_pct, (int, float)):
            last_hp_pct[name] = hp_pct
    avg_final_hp = sum(pokemon_p1[name] * last_hp_pct[name] for name in pokemon_p1) / len(pokemon_p1)
    if avg_initial_hp == 0:
        return 0.0
    return avg_final_hp / avg_initial_hp

def hp_variation_ratio_p2(battle, pokemon_stats, types):
    p2_team = reconstruct_p2_team(battle, pokemon_stats, types)
    pokemon_p2 = {name.lower(): stats["base_hp"] for name, stats in p2_team.items() if stats["base_hp"] > 0}
    last_hp_pct = {name: 1.0 for name in pokemon_p2.keys()}
    for turn in battle.get("battle_timeline", []):
        p2_state = turn.get("p2_pokemon_state", {})
        if not p2_state:
            continue
        name = p2_state.get("name", "").lower().strip()
        hp_pct = p2_state.get("hp_pct")
        if name in last_hp_pct:
            last_hp_pct[name] = hp_pct
    avg_initial_hp = sum(pokemon_p2.values()) / len(pokemon_p2)
    avg_final_hp = sum(pokemon_p2[name] * last_hp_pct[name] for name in pokemon_p2) / len(pokemon_p2)
    return avg_final_hp/avg_initial_hp

def difference_hp_ratio(battle,pokemon_stats,types):
    return  hp_variation_ratio_p1(battle)-hp_variation_ratio_p2(battle, pokemon_stats, types)

def attack_mean_p1(battle):
    total = 0
    for pokemon in battle["p1_team_details"]:
        total += int(pokemon["base_atk"])
    return float(total / 6)

def defense_mean_p1(battle):
    total = 0
    for pokemon in battle["p1_team_details"]:
        total += int(pokemon["base_def"])
    return float(total / 6)

def special_attack_mean_p1(battle):
    total = 0
    for pokemon in battle["p1_team_details"]:
        total += int(pokemon["base_spa"])
    return float(total / 6)

def special_defense_mean_p1(battle):
    total = 0
    for pokemon in battle["p1_team_details"]:
        total += int(pokemon["base_spd"])
    return float(total / 6)

def speed_mean_p1(battle):
    total = 0
    for pokemon in battle["p1_team_details"]:
        total += int(pokemon["base_spe"])
    return float(total / 6)

def hp_mean_p1(battle):
    total = 0
    for pokemon in battle["p1_team_details"]:
        total += int(pokemon["base_hp"])
    return float(total / 6)

def attack_mean_p2(battle, pokemon_stats, types):
    p2_team = reconstruct_p2_team(battle, pokemon_stats, types)
    if not p2_team:
        return 0.0
    total = sum(int(p["base_atk"]) for p in p2_team.values() if p.get("base_atk", 0) > 0)
    return float(total / len(p2_team))


def defense_mean_p2(battle, pokemon_stats, types):
    p2_team = reconstruct_p2_team(battle, pokemon_stats, types)
    if not p2_team:
        return 0.0
    total = sum(int(p["base_def"]) for p in p2_team.values() if p.get("base_def", 0) > 0)
    return float(total / len(p2_team))


def special_attack_mean_p2(battle, pokemon_stats, types):
    p2_team = reconstruct_p2_team(battle, pokemon_stats, types)
    if not p2_team:
        return 0.0
    total = sum(int(p["base_spa"]) for p in p2_team.values() if p.get("base_spa", 0) > 0)
    return float(total / len(p2_team))


def special_defense_mean_p2(battle, pokemon_stats, types):
    p2_team = reconstruct_p2_team(battle, pokemon_stats, types)
    if not p2_team:
        return 0.0
    total = sum(int(p["base_spd"]) for p in p2_team.values() if p.get("base_spd", 0) > 0)
    return float(total / len(p2_team))


def speed_mean_p2(battle, pokemon_stats, types):
    p2_team = reconstruct_p2_team(battle, pokemon_stats, types)
    if not p2_team:
        return 0.0
    total = sum(int(p["base_spe"]) for p in p2_team.values() if p.get("base_spe", 0) > 0)
    return float(total / len(p2_team))


def hp_mean_p2(battle, pokemon_stats, types):
    p2_team = reconstruct_p2_team(battle, pokemon_stats, types)
    if not p2_team:
        return 0.0
    total = sum(int(p["base_hp"]) for p in p2_team.values() if p.get("base_hp", 0) > 0)
    return float(total / len(p2_team))

def attack_diff(battle, pokemon_stats, types):
    return attack_mean_p1(battle) - attack_mean_p2(battle, pokemon_stats, types)

def defense_diff(battle, pokemon_stats, types):
    return defense_mean_p1(battle) - defense_mean_p2(battle, pokemon_stats, types)

def special_attack_diff(battle, pokemon_stats, types):
    return special_attack_mean_p1(battle) - special_attack_mean_p2(battle, pokemon_stats, types)

def special_defense_diff(battle, pokemon_stats, types):
    return special_defense_mean_p1(battle) - special_defense_mean_p2(battle, pokemon_stats, types)

def speed_diff(battle, pokemon_stats, types):
    return speed_mean_p1(battle) - speed_mean_p2(battle, pokemon_stats, types)

def hp_diff(battle, pokemon_stats, types):
    return hp_mean_p1(battle) - hp_mean_p2(battle, pokemon_stats, types)

def total_stat_diff(battle, pokemon_stats, types):
    diffs = [
        attack_diff(battle, pokemon_stats, types),
        defense_diff(battle, pokemon_stats, types),
        special_attack_diff(battle, pokemon_stats, types),
        special_defense_diff(battle, pokemon_stats, types),
        speed_diff(battle, pokemon_stats, types),
        hp_diff(battle, pokemon_stats, types)
    ]
    return float(sum(diffs) / len(diffs))

def hp_momentum(battle):
    p1_hp = []
    p2_hp = []
    for turn in battle["battle_timeline"]:
        if "p1_pokemon_state" in turn and "p2_pokemon_state" in turn:
            p1_hp.append(turn["p1_pokemon_state"]["hp_pct"])
            p2_hp.append(turn["p2_pokemon_state"]["hp_pct"])
    if not p1_hp or not p2_hp:
        return 0
    diffs = [p1_hp[i] - p2_hp[i] for i in range(len(p1_hp))]
    return sum(np.sign(np.diff(diffs)) > 0) / len(diffs)

def resistence_offensive_difference(battle,attack_types,types):
    p2=p2_team(battle,types)
    p1={}
    for pokemon in battle["p1_team_details"]:
        if isinstance(pokemon["types"],str):
            p1[pokemon["name"]]=pokemon["types"]
        if isinstance(pokemon["types"],list):
            pokemon_types=[]
            for i in range(len(pokemon["types"])):
                if pokemon["types"][i]=="notype":
                    continue
                pokemon_types.append(pokemon["types"][i])
            p1[pokemon["name"]]=pokemon_types
    resistence_p1=0
    offensive_p1=0
    resistence_p2=0
    offensive_p2=0
    types_p1=[]
    types_p2=[]
    types=p1.values()
    for type_ in types:
        if isinstance(type_,str):
            types_p1.append(type_)
        if isinstance(type_,list):
            for i in range(len(type_)):
                types_p1.append(type_[i])
    types_p1=set(types_p1)
    for type_ in types:
        if isinstance(type_,str):
            types_p2.append(type_)
        if isinstance(type_,list):
            for i in range(len(type_)):
                types_p2.append(type_[i])
    for type_ in types_p1:
        if type_ in attack_types:
            for def_type, mult in attack_types[type_].items():
                if def_type in types_p2:
                    if mult == 2:
                        offensive_p1 += 1
                    elif mult == 0.5 or mult==0:
                        resistence_p2 += 1 
    for type_ in types_p2:
        if type_ in attack_types:
            for def_type, mult in attack_types[type_].items():
                if def_type in types_p1:
                    if mult == 2:
                        offensive_p2 += 1
                    elif mult == 0.5 or mult==0:
                        resistence_p1 += 1
                        
    score_p1=offensive_p1+resistence_p1
    score_p2=offensive_p2+resistence_p2
    score=score_p1-score_p2
            
    return score

def variance_difference(battle,pokemon_stats,types):
    totals_p1 = []
    for pokemon in battle["p1_team_details"]:
        total = (
            pokemon.get("base_hp", 0)
            + pokemon.get("base_atk", 0)
            + pokemon.get("base_def", 0)
            + pokemon.get("base_spa", 0)
            + pokemon.get("base_spd", 0)
            + pokemon.get("base_spe", 0)
        )
        totals_p1.append(total)
    p1_var=float(np.var(totals_p1))
    p2=reconstruct_p2_team(battle, pokemon_stats, types)
    totals_p2=[]
    for pokemon in p2.values():
        total = (
            pokemon.get("base_hp", 0)
            + pokemon.get("base_atk", 0)
            + pokemon.get("base_def", 0)
            + pokemon.get("base_spa", 0)
            + pokemon.get("base_spd", 0)
            + pokemon.get("base_spe", 0)
        )
        totals_p2.append(total)
    p2_var=float(np.var(totals_p2))
    return p1_var-p2_var

def lead_difference(battle, attack_types):
    lead_p1 = battle["p1_team_details"][0]["name"].lower()
    lead_p2 = battle["p2_lead_details"]["name"].lower()
    types_p1 = [t.lower() for t in battle["p1_team_details"][0]["types"] if t != "notype"]
    types_p2 = [t.lower() for t in battle["p2_lead_details"]["types"] if t != "notype"]

    score = 0.0

    
    for atk_type in types_p1:
        for def_type in types_p2:
            multiplier = attack_types.get(atk_type, {}).get(def_type, 1.0)
            if multiplier > 1:        
                score += 1
            elif 0 < multiplier < 1: 
                score -= 1

   
    for atk_type in types_p2:
        for def_type in types_p1:
            multiplier = attack_types.get(atk_type, {}).get(def_type, 1.0)
            if multiplier > 1:
                score -= 1            
            elif 0 < multiplier < 1:
                score += 1            

    return score

def effective_status_p1(battle):
    crowd_control_status=["par","frz","slp"]
    count=0
    timeline = battle["battle_timeline"]
    battle_len = len(timeline)
    for turn in timeline:
        if turn["p1_pokemon_state"]["status"] in crowd_control_status:
            if turn["p1_move_details"] is None:
                if turn["turn"] < battle_len:
                    next_turn = timeline[turn["turn"]]
                    if turn["p1_pokemon_state"]["name"]==next_turn["p1_pokemon_state"]["name"]:
                        count+=1
    return(count/battle_len)

def effective_status_p2(battle):
    crowd_control_status=["par","frz","slp"]
    count=0
    timeline = battle["battle_timeline"]
    battle_len = len(timeline)
    for turn in timeline:
        if turn["p2_pokemon_state"]["status"] in crowd_control_status:
            if turn["p2_move_details"] is None:
                if turn["turn"] < battle_len:
                    next_turn = timeline[turn["turn"]]
                    if turn["p2_pokemon_state"]["name"]==next_turn["p2_pokemon_state"]["name"]:
                        count+=1
    return(count/battle_len)

def effective_status_diff(battle):
    p1=effective_status_p1(battle)
    p2=effective_status_p2(battle)
    return p2-p1
