"""
generate_messy_housing_data.py
==============================
Interactive Australian housing market messy data generator.

Asks the user which years and months they want, then produces data whose
prices, auction clearance rates, days-on-market, listing volumes, and
market commentary all reflect REAL conditions for that period:

  2020  – COVID crash then stimulus rebound; RBA cuts to 0.10%; Melbourne
           262-day lockdown suppresses auctions; regional tree-change surge
  2021  – Pandemic boom; prices +24.5% nationally; FOMO; ultra-low rates;
           Melbourne lockdowns then explosive post-lockdown release
  2022  – RBA hikes 10x (0.10% → 3.10%); Sydney/Melbourne prices fall;
           clearance rates collapse; APRA tightens lending; sentiment sours
  2023  – Surprise recovery despite 4.35% peak rate; supply crunch;
           population boom; prices +8.1% nationally; Perth/Brisbane surge
  2024  – RBA holds at 4.35% all year then Dec cut to 4.10%; Melbourne
           flat/weak; affordability stress; days-on-market rise
  2025  – Three cuts (4.35%→3.60%); renewed confidence; prices +8.6%;
           Melbourne edges above prior peak; Adelaide/Perth strong

Run:
    python generate_messy_housing_data.py
"""

import os
import sys
import re
import random
import calendar
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Market conditions lookup – every month from Jan 2020 to Dec 2025
# ─────────────────────────────────────────────────────────────────────────────
# Each entry covers one calendar period and carries:
#   rba_rate        – RBA cash rate in effect (%)
#   price_index     – multiplier on base suburb median (1.0 = baseline Jan 2020)
#   clearance_mel   – Melbourne auction clearance rate (%)
#   clearance_syd   – Sydney auction clearance rate (%)
#   dom_delta       – days-on-market shift (+/- days vs. typical)
#   vol_factor      – listing volume relative to normal (1.0 = normal)
#   sentiment       – buyer sentiment label (used in notes column)
#   notes           – one-line market context written to the 'market_context' column

def _mc(rba, pidx, clr_mel, clr_syd, dom, vol, sent, note):
    return dict(rba_rate=rba, price_index=pidx, clearance_mel=clr_mel,
                clearance_syd=clr_syd, dom_delta=dom, vol_factor=vol,
                sentiment=sent, notes=note)

MARKET_CONDITIONS = {
    # ── 2020 ──────────────────────────────────────────────────────────────
    (2020,  1): _mc(0.75, 1.000, 64, 68, +5,  1.0, "Cautious",    "Pre-COVID; bushfire smoke over cities; cautious buyer sentiment"),
    (2020,  2): _mc(0.75, 1.002, 65, 70, +4,  1.0, "Cautious",    "RBA signals concern; bushfire recovery; market steady"),
    (2020,  3): _mc(0.50, 0.985, 30, 35, +18, 0.4, "Panic",       "COVID-19 declared pandemic; RBA emergency cut to 0.50%; auctions banned in VIC/NSW"),
    (2020,  4): _mc(0.25, 0.972, 22, 28, +25, 0.3, "Panic",       "National lockdown; auction volumes collapse; RBA cuts to 0.25%; JobKeeper announced"),
    (2020,  5): _mc(0.25, 0.968, 32, 38, +20, 0.4, "Fearful",     "Virtual auctions begin; prices soften; JobKeeper cushions market"),
    (2020,  6): _mc(0.25, 0.971, 45, 52, +12, 0.5, "Cautious",    "VIC eases; Sydney begins recovery; government stimulus supports sentiment"),
    (2020,  7): _mc(0.25, 0.965, 38, 58, +15, 0.5, "Mixed",       "Melbourne re-enters Stage 4 lockdown; Sydney recovers strongly"),
    (2020,  8): _mc(0.25, 0.963, 35, 62, +16, 0.4, "Mixed",       "Melbourne Stage 4 peak; curfew in place; Sydney prices lift"),
    (2020,  9): _mc(0.25, 0.967, 42, 65, +12, 0.5, "Cautious",    "Melbourne lockdown easing signals; regional markets surge (tree-change)"),
    (2020, 10): _mc(0.25, 0.978, 58, 70, +8,  0.7, "Improving",   "Melbourne reopening; pent-up demand releases; HomeBuilder boost"),
    (2020, 11): _mc(0.10, 0.992, 66, 74, +5,  0.9, "Positive",    "RBA cuts to record 0.10%; first home buyer surge; clearance rates recover"),
    (2020, 12): _mc(0.10, 1.005, 70, 76, +3,  0.8, "Positive",    "Year-end confidence; FHLDS expanded; regional/coastal markets boom"),

    # ── 2021 ──────────────────────────────────────────────────────────────
    (2021,  1): _mc(0.10, 1.025, 72, 79, +0,  1.1, "Confident",   "Pandemic boom begins; ultra-low rates; FOMO sets in; supply tight"),
    (2021,  2): _mc(0.10, 1.048, 75, 82, -2,  1.2, "Strong",      "Prices accelerating; HomeBuilder deadline rush; CoreLogic records +2.1% monthly"),
    (2021,  3): _mc(0.10, 1.073, 76, 84, -3,  1.3, "FOMO",        "Strongest monthly growth in 32 years; buyers waiving conditions"),
    (2021,  4): _mc(0.10, 1.098, 77, 85, -4,  1.3, "FOMO",        "Melbourne median hits $900k+; regional prices up 15%+; investor surge"),
    (2021,  5): _mc(0.10, 1.118, 76, 83, -4,  1.2, "FOMO",        "Prices +8% YTD; APRA warns on lending; short lockdowns in VIC/QLD"),
    (2021,  6): _mc(0.10, 1.135, 68, 80, -2,  1.1, "Strong",      "Victoria snap lockdown (June); clearance dips briefly; national momentum holds"),
    (2021,  7): _mc(0.10, 1.148, 60, 75, +3,  1.0, "Mixed",       "Melbourne lockdown #5; Sydney Delta lockdown begins; auction volumes drop"),
    (2021,  8): _mc(0.10, 1.155, 55, 72, +5,  0.9, "Anxious",     "Sydney 100-day lockdown peak; Melbourne lockdown #6; nationwide uncertainty"),
    (2021,  9): _mc(0.10, 1.162, 58, 78, +4,  0.9, "Recovering",  "NSW roadmap out of lockdown announced; confidence begins recovering"),
    (2021, 10): _mc(0.10, 1.175, 72, 83, -2,  1.2, "Strong",      "Melbourne 262-day lockdown ends; explosive post-lockdown release; record auctions"),
    (2021, 11): _mc(0.10, 1.185, 76, 84, -3,  1.3, "FOMO",        "Melbourne median hits record $1,094k; Sydney above $1.3M; national +24.5% YoY"),
    (2021, 12): _mc(0.10, 1.190, 74, 82, -2,  1.0, "Strong",      "Year closes near peak; Omicron emerges but market shrugs it off"),

    # ── 2022 ──────────────────────────────────────────────────────────────
    (2022,  1): _mc(0.10, 1.188, 72, 80, -1,  1.1, "Ebullient",   "Market still hot; RBA signals no early hikes; prices at or near peak"),
    (2022,  2): _mc(0.10, 1.183, 70, 78, +0,  1.1, "Cooling",     "Rate hike expectations building; APRA buffer at 3%; early signs of softening"),
    (2022,  3): _mc(0.10, 1.172, 66, 74, +3,  1.0, "Cautious",    "Ukraine war drives global inflation; RBA under pressure; Sydney falls 0.2%"),
    (2022,  4): _mc(0.10, 1.158, 62, 69, +5,  0.9, "Cautious",    "Sydney -0.4% in April; Melbourne softening; first rate hike imminent"),
    (2022,  5): _mc(0.35, 1.140, 56, 63, +8,  0.9, "Turning",     "RBA hikes 25bp to 0.35% — first hike in 11 years; clearance rates fall sharply"),
    (2022,  6): _mc(0.85, 1.112, 50, 57, +12, 0.8, "Falling",     "RBA hikes 50bp to 0.85%; Sydney -1.0% in month; Melbourne -0.7%; panic sets in"),
    (2022,  7): _mc(1.35, 1.082, 46, 52, +15, 0.8, "Falling",     "RBA hikes 50bp to 1.35%; prices falling across all capitals; listings rise"),
    (2022,  8): _mc(1.85, 1.055, 44, 49, +17, 0.8, "Distressed",  "RBA hikes 50bp to 1.85%; borrowing capacity down 25%; forced sales rising"),
    (2022,  9): _mc(2.35, 1.030, 43, 48, +18, 0.9, "Distressed",  "RBA hikes 50bp to 2.35%; affordability at historic lows; sentiment very negative"),
    (2022, 10): _mc(2.60, 1.012, 44, 50, +16, 0.9, "Weak",        "RBA slows to 25bp hikes; prices still falling but pace eases"),
    (2022, 11): _mc(2.85, 0.998, 46, 52, +14, 0.9, "Weak",        "National prices -6% from peak; Melbourne -7%; clearance slowly recovering"),
    (2022, 12): _mc(3.10, 0.988, 52, 58, +12, 0.7, "Cautious",    "RBA signals slower pace; year closes with prices well off 2021 peak"),

    # ── 2023 ──────────────────────────────────────────────────────────────
    (2023,  1): _mc(3.10, 0.985, 55, 62, +10, 0.8, "Cautious",    "Surprise: prices stabilise; population surge; supply still constrained"),
    (2023,  2): _mc(3.35, 0.990, 58, 65, +8,  0.9, "Stabilising", "RBA hikes 25bp; but prices begin recovering against expectations"),
    (2023,  3): _mc(3.60, 0.998, 62, 69, +6,  1.0, "Recovering",  "Prices tick up nationally; migration at record highs; rental crisis deepens"),
    (2023,  4): _mc(3.85, 1.010, 65, 72, +4,  1.0, "Recovering",  "Perth +3% in April; Brisbane strong; Melbourne lagging other capitals"),
    (2023,  5): _mc(3.85, 1.022, 66, 73, +3,  1.1, "Improving",   "RBA pauses; market reads it as peak; buyer confidence returns"),
    (2023,  6): _mc(4.10, 1.032, 64, 71, +4,  1.1, "Resilient",   "Surprise RBA hike to 4.10%; prices still rising on supply shortage"),
    (2023,  7): _mc(4.10, 1.042, 65, 72, +3,  1.1, "Resilient",   "CoreLogic: +5% nationally in 2023 so far; Perth +12%; Brisbane +9%"),
    (2023,  8): _mc(4.10, 1.050, 65, 71, +4,  1.0, "Resilient",   "Affordability stress growing; first-home buyers retreating"),
    (2023,  9): _mc(4.10, 1.056, 64, 70, +4,  1.0, "Resilient",   "Labor announces Help to Buy scheme; overseas migration at record 500k"),
    (2023, 10): _mc(4.10, 1.059, 63, 69, +5,  1.0, "Steady",      "Market holding; listing volumes rising in Sydney/Melbourne"),
    (2023, 11): _mc(4.35, 1.056, 62, 68, +6,  1.0, "Cautious",    "RBA final hike to 4.35%; clearance dips on rate shock; sentiment softens"),
    (2023, 12): _mc(4.35, 1.058, 66, 72, +5,  0.8, "Cautious",    "Year closes +8.1% nationally; below 2021 but well above 2022 trough"),

    # ── 2024 ──────────────────────────────────────────────────────────────
    (2024,  1): _mc(4.35, 1.055, 60, 66, +7,  0.9, "Subdued",     "RBA holds; Melbourne falls 1.5% Q1; affordability at record lows"),
    (2024,  2): _mc(4.35, 1.050, 59, 65, +8,  0.9, "Subdued",     "Rate-cut hopes pushed to late 2024; household budgets under stress"),
    (2024,  3): _mc(4.35, 1.048, 60, 66, +7,  1.0, "Subdued",     "Melbourne median $1,032k (flat YoY); other capitals outperforming"),
    (2024,  4): _mc(4.35, 1.051, 61, 67, +7,  1.0, "Stabilising", "Supply rising in Melbourne and Sydney; prices broadly flat"),
    (2024,  5): _mc(4.35, 1.055, 62, 68, +6,  1.0, "Stabilising", "CoreLogic: national +0.6% monthly; Perth/Brisbane remain hot"),
    (2024,  6): _mc(4.35, 1.058, 61, 67, +7,  1.0, "Steady",      "Budget stimulus for first home buyers; Help to Buy passes Senate"),
    (2024,  7): _mc(4.35, 1.055, 60, 65, +8,  1.0, "Cautious",    "Rate-cut timing uncertain; buyers sitting on the fence"),
    (2024,  8): _mc(4.35, 1.057, 61, 66, +7,  1.0, "Cautious",    "Strong employment keeps RBA on hold; cost-of-living still a concern"),
    (2024,  9): _mc(4.35, 1.060, 62, 67, +6,  1.0, "Steady",      "Spring selling season; listings up; days on market rising in Melbourne"),
    (2024, 10): _mc(4.35, 1.063, 63, 68, +6,  1.1, "Steady",      "Clearance rates holding; prices up modestly; investors return cautiously"),
    (2024, 11): _mc(4.35, 1.060, 62, 67, +7,  1.1, "Cautious",    "RBA Nov meeting: hold confirmed; rate-cut expectations for Feb 2025 grow"),
    (2024, 12): _mc(4.10, 1.065, 68, 74, +4,  0.8, "Optimistic",  "RBA cuts 25bp to 4.10%; buyer confidence rebounds; year ends on positive note"),

    # ── 2025 ──────────────────────────────────────────────────────────────
    (2025,  1): _mc(4.10, 1.072, 66, 72, +4,  1.0, "Improving",   "Rate-cut cycle underway; confidence improving; Adelaide and Perth surge"),
    (2025,  2): _mc(3.85, 1.082, 68, 74, +3,  1.1, "Positive",    "RBA cuts to 3.85%; borrowing capacity rises ~$25k; listings up in spring markets"),
    (2025,  3): _mc(3.85, 1.092, 69, 75, +2,  1.1, "Positive",    "Melbourne edges above prior 2022 peak; clearance rates rising"),
    (2025,  4): _mc(3.85, 1.100, 70, 76, +1,  1.2, "Strong",      "Renewed FOMO in some markets; regional VIC recovering; investor activity up"),
    (2025,  5): _mc(3.60, 1.112, 72, 78, +0,  1.2, "Strong",      "RBA cuts to 3.60%; national prices +5.5% YTD; sentiment approaching 2021 levels"),
    (2025,  6): _mc(3.60, 1.120, 73, 79, -1,  1.2, "Strong",      "Supply still tight; Help to Buy demand; Brisbane median crosses $1M"),
    (2025,  7): _mc(3.60, 1.128, 72, 78, -1,  1.1, "Strong",      "Inflation remains above target; RBA signals caution on further cuts"),
    (2025,  8): _mc(3.60, 1.135, 71, 77, +1,  1.1, "Positive",    "RBA holds; CoreLogic: +8.6% nationally for year to Aug; record national median"),
    (2025,  9): _mc(3.60, 1.140, 70, 76, +2,  1.1, "Steady",      "Spring season; listings rise; Melbourne middle-ring suburbs strongest"),
    (2025, 10): _mc(3.60, 1.145, 70, 76, +2,  1.1, "Steady",      "National median approaches $900k; affordability concerns return"),
    (2025, 11): _mc(3.60, 1.148, 71, 77, +1,  1.1, "Positive",    "RBA holds at 3.60%; market confident; rate-hike risk emerging late in year"),
    (2025, 12): _mc(3.60, 1.152, 75, 80, -1,  0.9, "Confident",   "Year closes +8.6%; record national median $901k; headwinds building for 2026"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Reference data
# ─────────────────────────────────────────────────────────────────────────────

SUBURBS = [
    # (suburb, council, postcode, state, base_house_median_jan2020, region, cbd_dist_km)
    ("Richmond",      "Yarra",             3121, "VIC", 1_250_000, "Inner Melbourne",      3.0),
    ("Fitzroy",       "Yarra",             3065, "VIC", 1_100_000, "Inner Melbourne",      2.5),
    ("Collingwood",   "Yarra",             3066, "VIC",   990_000, "Inner Melbourne",      2.8),
    ("Brunswick",     "Moreland",          3056, "VIC",   870_000, "Northern Metropolitan",5.5),
    ("Northcote",     "Darebin",           3070, "VIC",   870_000, "Northern Metropolitan",6.5),
    ("Preston",       "Darebin",           3072, "VIC",   740_000, "Northern Metropolitan",9.0),
    ("Prahran",       "Stonnington",       3181, "VIC", 1_050_000, "Inner South",          4.5),
    ("South Yarra",   "Stonnington",       3141, "VIC", 1_400_000, "Inner South",          3.5),
    ("Toorak",        "Stonnington",       3142, "VIC", 3_200_000, "Inner East",           6.0),
    ("Hawthorn",      "Boroondara",        3122, "VIC", 1_600_000, "Inner East",           6.5),
    ("Camberwell",    "Boroondara",        3124, "VIC", 1_550_000, "Inner East",           9.5),
    ("St Kilda",      "Port Phillip",      3182, "VIC",   900_000, "Inner South",          5.5),
    ("Brighton",      "Bayside",           3186, "VIC", 2_100_000, "Southern Metropolitan",12.0),
    ("Footscray",     "Maribyrnong",       3011, "VIC",   720_000, "Western Metropolitan", 6.0),
    ("Sunshine",      "Brimbank",          3020, "VIC",   590_000, "Western Metropolitan",12.5),
    ("Werribee",      "Wyndham",           3030, "VIC",   480_000, "Western Metropolitan",30.0),
    ("Craigieburn",   "Hume",              3064, "VIC",   530_000, "Northern Metropolitan",26.0),
    ("Epping",        "Whittlesea",        3076, "VIC",   580_000, "Northern Metropolitan",22.0),
    ("Box Hill",      "Whitehorse",        3128, "VIC",   990_000, "Eastern Metropolitan", 14.0),
    ("Doncaster",     "Manningham",        3108, "VIC", 1_050_000, "Eastern Metropolitan", 14.5),
    ("Glen Waverley", "Monash",            3150, "VIC",   960_000, "Eastern Metropolitan", 22.0),
    ("Ringwood",      "Maroondah",         3134, "VIC",   770_000, "Eastern Metropolitan", 24.0),
    ("Frankston",     "Frankston",         3199, "VIC",   550_000, "Southern Metropolitan",40.0),
    ("Dandenong",     "Greater Dandenong", 3175, "VIC",   540_000, "Eastern Metropolitan", 30.0),
    ("Geelong",       "Greater Geelong",   3220, "VIC",   530_000, "Regional VIC",         75.0),
    ("Ballarat",      "City of Ballarat",  3350, "VIC",   390_000, "Regional VIC",        115.0),
    ("Bendigo",       "Greater Bendigo",   3550, "VIC",   375_000, "Regional VIC",        150.0),
    ("Pakenham",      "Cardinia",          3810, "VIC",   530_000, "Eastern Metropolitan", 55.0),
    ("Officer",       "Cardinia",          3809, "VIC",   520_000, "Eastern Metropolitan", 50.0),
    ("Sunbury",       "Hume",              3429, "VIC",   510_000, "Northern Metropolitan",40.0),
    # NSW
    ("Surry Hills",   "Sydney",            2010, "NSW", 1_350_000, "Inner Sydney",          3.0),
    ("Newtown",       "Inner West",        2042, "NSW", 1_250_000, "Inner Sydney",          5.0),
    ("Parramatta",    "Parramatta",        2150, "NSW",   820_000, "Western Sydney",       23.0),
    ("Blacktown",     "Blacktown",         2148, "NSW",   710_000, "Western Sydney",       33.0),
    ("Bondi",         "Waverley",          2026, "NSW", 2_200_000, "Inner Sydney",          7.0),
    ("Manly",         "Northern Beaches",  2095, "NSW", 2_800_000, "Inner Sydney",         17.0),
    ("Chatswood",     "Willoughby",        2067, "NSW", 2_000_000, "Inner Sydney",         10.0),
    ("Penrith",       "Penrith",           2750, "NSW",   680_000, "Western Sydney",       52.0),
    ("Liverpool",     "Liverpool",         2170, "NSW",   680_000, "Western Sydney",       30.0),
    # QLD
    ("Paddington",    "Brisbane",          4064, "QLD",   890_000, "Brisbane Inner",        3.5),
    ("Sunnybank",     "Brisbane",          4109, "QLD",   680_000, "Brisbane Outer",       14.0),
    ("Ipswich",       "Ipswich",           4305, "QLD",   440_000, "South-East QLD",       40.0),
    ("Gold Coast",    "Gold Coast",        4217, "QLD",   750_000, "Gold Coast",           80.0),
    ("Sunshine Coast","Sunshine Coast",    4557, "QLD",   720_000, "Sunshine Coast",      100.0),
    ("Toowoomba",     "Toowoomba",         4350, "QLD",   380_000, "Regional QLD",        130.0),
    # WA
    ("Subiaco",       "Subiaco",           6008, "WA",    980_000, "Inner Perth",           4.0),
    ("Fremantle",     "Fremantle",         6160, "WA",    820_000, "Inner Perth",          19.0),
    ("Cottesloe",     "Cottesloe",         6011, "WA",  1_650_000, "Inner Perth",          11.0),
    ("Northbridge",   "Perth",             6003, "WA",    680_000, "Inner Perth",           2.0),
    ("Joondalup",     "Joondalup",         6027, "WA",    570_000, "Northern Perth",       26.0),
    ("Mandurah",      "Mandurah",          6210, "WA",    410_000, "Regional WA",          74.0),
    ("Baldivis",      "Rockingham",        6171, "WA",    490_000, "Southern Perth",       47.0),
    ("Ellenbrook",    "Swan",              6069, "WA",    520_000, "North Eastern Perth",  27.0),
    ("Applecross",    "Melville",          6153, "WA",  1_380_000, "Southern Perth",        9.0),
    ("Bunbury",       "Bunbury",           6230, "WA",    380_000, "Regional WA",         175.0),
    # SA
    ("Norwood",       "Norwood Payneham St Peters", 5067, "SA", 1_050_000, "Inner Adelaide",  4.0),
    ("Glenelg",       "Holdfast Bay",      5045, "SA",    850_000, "Coastal Adelaide",     11.0),
    ("Unley",         "Unley",             5061, "SA",  1_100_000, "Inner Adelaide",        5.0),
    ("Prospect",      "Prospect",          5082, "SA",    780_000, "Inner Adelaide",        5.0),
    ("Tea Tree Gully","Tea Tree Gully",    5091, "SA",    580_000, "North Eastern Adelaide",18.0),
    ("Mount Barker",  "Mount Barker",      5251, "SA",    530_000, "Regional SA",          35.0),
    ("Port Adelaide",  "Port Adelaide Enfield", 5015, "SA", 620_000, "North Western Adelaide", 14.0),
    # TAS
    ("Sandy Bay",     "Hobart",            7005, "TAS",   780_000, "Hobart Inner",          4.0),
    ("Launceston",    "Launceston",        7250, "TAS",   430_000, "Regional TAS",        198.0),
    ("Glenorchy",     "Glenorchy",         7010, "TAS",   490_000, "Hobart Greater",        9.0),
    ("Devonport",     "Devonport",         7310, "TAS",   360_000, "Regional TAS",        280.0),
    # ACT
    ("Braddon",       "City of Canberra",  2612, "ACT",   850_000, "Inner Canberra",        3.0),
    ("Gungahlin",     "City of Canberra",  2912, "ACT",   730_000, "Northern Canberra",    14.0),
    ("Tuggeranong",   "City of Canberra",  2900, "ACT",   620_000, "Southern Canberra",    16.0),
    ("Woden Valley",  "City of Canberra",  2606, "ACT",   690_000, "Southern Canberra",     8.0),
    # NT
    ("Darwin CBD",    "Darwin",            "0800", "NT",    530_000, "Darwin Inner",          1.0),
    ("Palmerston",    "Palmerston",        "0830", "NT",    420_000, "Darwin Greater",        22.0),
    ("Alice Springs", "Alice Springs",     "0870", "NT",    380_000, "Regional NT",         1500.0),
]

PROPERTY_TYPES = ["House", "Unit", "Townhouse", "Apartment", "Villa", "Duplex", "Land"]

PROPERTY_TYPE_VARIANTS = {
    "House":     ["House", "house", "HOUSE", "h", "House/Cottage", "Residential House", "house ", "Hse"],
    "Unit":      ["Unit", "unit", "UNIT", "u", "Unit/Duplex", "unit duplex", "UNIT "],
    "Townhouse": ["Townhouse", "townhouse", "t", "Town House", "TownHouse", "TOWNHOUSE"],
    "Apartment": ["Apartment", "apartment", "apt", "APT", "Flat", "flat"],
    "Villa":     ["Villa", "villa", "VILLA", "Villa Unit"],
    "Duplex":    ["Duplex", "duplex", "Semi-Detached", "semi"],
    "Land":      ["Land", "land", "Vacant Land", "vacant land", "dev site"],
}

SALE_METHODS = {
    "Auction":        ["Auction", "auction", "AUC", "A"],
    "Private Sale":   ["Private Sale", "private sale", "S", "Private Treaty", "PS"],
    "EOI":            ["EOI", "Expression of Interest", "eoi"],
    "Passed In":      ["Passed In", "PI", "passed in"],
    "Tender":         ["Tender", "tender", "TEN"],
}

AGENTS = [
    "Ray White", "Barry Plant", "Hocking Stuart", "Jellis Craig",
    "Marshall White", "McGrath", "LJ Hooker", "Nelson Alexander",
    "Woodards", "Biggin & Scott", "Buxton", "RT Edgar",
    "Raine & Horne", "First National", "Fletchers", "Noel Jones",
    "Kay & Burton", "Stockdale & Leggo", "Century 21", "PRD",
]

STREET_NAMES = ["Smith", "Jones", "Brown", "Oak", "Main", "Church", "Park",
                "Station", "High", "Victoria", "King", "Queen", "Railway"]
STREET_TYPES = ["St", "Street", "Rd", "Road", "Ave", "Avenue", "Cres", "Dr",
                "Drive", "Pde", "Parade", "Ct", "Court", "Pl", "Place"]

rng = np.random.default_rng(0)  # overridden in main once periods are known


# ─────────────────────────────────────────────────────────────────────────────
# Mess helpers  (identical to original, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def maybe_null(value, rate):
    return np.nan if rng.random() < rate else value

def messy_suburb(name):
    r = rng.random()
    if r < 0.10: return name.upper()
    if r < 0.18: return name.lower()
    if r < 0.23: return name.lower().replace(" ", "_")
    if r < 0.27: return " " + name
    if r < 0.30: return name + " "
    return name

def messy_state(state):
    v = {
        "VIC": ["VIC","Vic","vic","Victoria","victoria","vic.","V.I.C"],
        "NSW": ["NSW","Nsw","nsw","New South Wales","N.S.W","nsw."],
        "QLD": ["QLD","Qld","qld","Queensland","Q'land","QLD."],
        "WA":  ["WA","Wa","wa","Western Australia","W.A","western australia"],
        "SA":  ["SA","Sa","sa","South Australia","S.A","south australia","S.A."],
        "TAS": ["TAS","Tas","tas","Tasmania","tasmania","Tassie","TAS."],
        "ACT": ["ACT","Act","act","Australian Capital Territory","A.C.T"],
        "NT":  ["NT","Nt","nt","Northern Territory","N.T","northern territory"],
    }
    return rng.choice(v.get(state, [state]))

def messy_date(d):
    fmt = rng.choice(["%d/%m/%Y","%Y-%m-%d","%d-%m-%Y","%d %b %Y",
                      "%d %B %Y","%b %d, %Y","%d/%m/%y","%b-%y","%Y/%m/%d"])
    return d.strftime(fmt)

def messy_price(price):
    r = rng.random()
    if r < 0.04: return "Contact Agent"
    if r < 0.07: return "POA"
    if r < 0.10: return "Offers Over $" + f"{int(price - price % 50_000):,}"
    if r < 0.14: return f"${price/1_000_000:.2f}M"
    if r < 0.18: return f"${price/1_000:.0f}k"
    if r < 0.26: return f"${price:,.0f}"
    if r < 0.33: return f"{price:.0f}"
    if r < 0.38: return f"${price:,.2f}"
    return f"${price:,}"

def messy_bedrooms(n):
    words = ["zero","one","two","three","four","five","six","seven","eight","nine","ten"]
    r = rng.random()
    if r < 0.05: return str(float(n))
    if r < 0.09: return words[min(n, 10)]
    if r < 0.13: return f"{n} bed"
    if r < 0.17: return f"{n} Bedrooms"
    if r < 0.19: return f"{n:02d}"
    if r < 0.22: return f"{n} BR"
    return str(n)

def messy_land_size(sqm):
    r = rng.random()
    if r < 0.12: return f"{sqm:.1f} sqm"
    if r < 0.22: return f"{sqm:.0f}m2"
    if r < 0.30: return f"{sqm/10000:.4f} ha"
    if r < 0.35: return f"{sqm:.0f} square metres"
    if r < 0.39: return f"{sqm/10000:.3f}ha"
    return str(int(sqm))

def messy_null():
    return rng.choice(["N/A","n/a","NA","-","--","unknown","Unknown","none","None"," ","?","TBC"])

def messy_bool(flag):
    if flag:
        return rng.choice(["Yes","yes","Y","True","true","1"])
    return rng.choice(["No","no","N","False","false","0"])


# ─────────────────────────────────────────────────────────────────────────────
# Price modelling: suburb base × period index × type discount × bedroom premium
# ─────────────────────────────────────────────────────────────────────────────

TYPE_MULTIPLIERS = {
    "House":     1.00,
    "Townhouse": 0.72,
    "Villa":     0.65,
    "Duplex":    0.78,
    "Unit":      0.55,
    "Apartment": 0.52,
    "Land":      0.40,
}

def compute_price(suburb_base, prop_type, bedrooms, mc):
    """
    Calculate a realistic price using:
      - suburb base median (Jan 2020 anchor)
      - CoreLogic-calibrated price_index for the exact month
      - property type multiplier
      - bedroom premium
      - lognormal noise
    """
    type_mult = TYPE_MULTIPLIERS.get(prop_type, 1.0)
    bed_premium = (bedrooms - 3) * suburb_base * 0.07
    base = (suburb_base * mc["price_index"] * type_mult) + bed_premium
    base = max(150_000, base)
    # Add realistic noise via lognormal (sigma=0.18 = ~18% std dev)
    noisy = base * float(np.exp(rng.normal(0, 0.18)))
    return round(noisy / 1_000) * 1_000


def inject_price_outlier(price):
    r = rng.random()
    if r < 0.012: return price * rng.uniform(15, 60)
    if r < 0.022: return rng.uniform(5_000, 45_000)
    return price


# ─────────────────────────────────────────────────────────────────────────────
# Auction method mix  –  shifts with market conditions
# ─────────────────────────────────────────────────────────────────────────────

def choose_sale_method(mc, state):
    """
    Higher clearance rate → more Auction listings.
    Falling market → more Passed In / Private Sale.
    """
    clearance = mc["clearance_mel"] if state == "VIC" else mc["clearance_syd"]
    # Auction share scales with clearance rate
    p_auction     = clearance / 200       # 65% clearance → 32.5% auction share
    p_passed_in   = max(0.02, (100 - clearance) / 400)
    p_private     = 1 - p_auction - p_passed_in - 0.05
    p_eoi         = 0.04
    p_tender      = 0.01
    total = p_auction + p_passed_in + p_private + p_eoi + p_tender
    probs = [p_auction/total, p_passed_in/total, p_private/total, p_eoi/total, p_tender/total]
    methods = ["Auction", "Passed In", "Private Sale", "EOI", "Tender"]
    return rng.choice(methods, p=probs)


# ─────────────────────────────────────────────────────────────────────────────
# Days on market  –  shifts with market conditions
# ─────────────────────────────────────────────────────────────────────────────

def compute_dom(mc, prop_type):
    base_dom = {"House": 28, "Townhouse": 32, "Unit": 35,
                "Apartment": 38, "Villa": 40, "Duplex": 36, "Land": 55}
    dom = base_dom.get(prop_type, 35) + mc["dom_delta"]
    dom = max(1, int(dom + rng.normal(0, 8)))
    return dom


# ─────────────────────────────────────────────────────────────────────────────
# COVID / market-event effects on specific fields
# ─────────────────────────────────────────────────────────────────────────────

def covid_inspection_note(year, month, state):
    """Inject lockdown-specific notes into address/source fields."""
    if state == "VIC":
        if (year == 2020 and month in [3,4,5]) or \
           (year == 2021 and month in [7,8,9]):
            return rng.choice([
                "Virtual inspection only",
                "Online auction",
                "By private appointment - lockdown restrictions apply",
                "Virtual tour available",
            ])
        if year == 2021 and month == 10:
            return rng.choice([
                "First open after lockdown",
                "High demand – multiple offers",
                "Post-lockdown release – inspect Saturday",
            ])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Row generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_row(row_id, year, month, suburb_pool):
    mc = MARKET_CONDITIONS[(year, month)]

    suburb_data = suburb_pool[rng.integers(0, len(suburb_pool))]
    suburb, council, postcode, state, base_price, region, cbd_dist = suburb_data

    # Regional VIC gets a tree-change premium 2020–2021
    if "Regional" in region and year in [2020, 2021]:
        base_price = int(base_price * rng.uniform(1.08, 1.18))

    prop_type = rng.choice(PROPERTY_TYPES, p=[0.45,0.20,0.15,0.08,0.05,0.04,0.03])
    is_land = prop_type == "Land"

    bedrooms   = int(rng.integers(1, 7)) if not is_land else 0
    bathrooms  = rng.choice([1,1,1,2,2,3]) if not is_land else 0
    car_spaces = rng.choice([0,1,1,2,2,3]) if not is_land else int(rng.integers(0, 3))
    toilets    = bathrooms + rng.choice([0,0,0,1])
    land_sqm   = float(rng.integers(120, 2000)) if not is_land else float(rng.integers(500, 50000))
    build_sqm  = float(rng.integers(60, 500)) if not is_land else np.nan

    yr_built_val = int(rng.integers(1900, year)) if not is_land else np.nan
    if yr_built_val is not np.nan and rng.random() < 0.018:
        yr_built_val = rng.choice([1066, 9999, 2099, 0])

    raw_price = compute_price(base_price, prop_type, bedrooms, mc)
    raw_price = inject_price_outlier(raw_price)

    # Pick a random date within the requested month
    last_day = calendar.monthrange(year, month)[1]
    sale_date = date(year, month, int(rng.integers(1, last_day + 1)))

    method_clean = choose_sale_method(mc, state)
    method_messy = rng.choice(SALE_METHODS.get(method_clean, [method_clean]))

    dom = compute_dom(mc, prop_type)
    agent = rng.choice(AGENTS)
    weekly_rent = round(raw_price * rng.uniform(0.028, 0.050) / 52 / 10) * 10 \
                  if not is_land else np.nan

    lockdown_note = covid_inspection_note(year, month, state)
    inspection_note = lockdown_note if lockdown_note else maybe_null(
        rng.choice(["Inspect Sat/Sun", "By appointment", "Open Saturday"]), 0.6)

    row = {
        # ── Identifiers ────────────────────────────────────────────────
        "listing_id":   f"AU-{row_id:06d}",
        "source":       rng.choice(["Domain","REA","Agent Website",
                                     "domain.com.au","realestate.com.au","Domain "," REA"]),

        # ── Location ──────────────────────────────────────────────────
        "address": maybe_null(
            f"{rng.integers(1,200)} {rng.choice(STREET_NAMES)} {rng.choice(STREET_TYPES)}",
            0.08),
        "suburb":        maybe_null(messy_suburb(suburb), 0.02),
        "state":         maybe_null(messy_state(state), 0.03),
        "postcode":      maybe_null(
            postcode if rng.random() > 0.06 else int(rng.integers(2000, 4999)), 0.04),
        "council_area":  maybe_null(
            council if rng.random() > 0.07 else council.upper(), 0.09),
        "region":        maybe_null(region, 0.12),
        "distance_to_cbd_km": maybe_null(
            cbd_dist if rng.random() > 0.05 else f"{cbd_dist} km", 0.10),
        "lat": maybe_null(round(rng.uniform(-38.5,-37.5) if state=="VIC"
                                else rng.uniform(-34.0,-33.5), 6), 0.25),
        "lon": maybe_null(round(rng.uniform(144.5,145.5) if state=="VIC"
                                else rng.uniform(150.5,151.5), 6), 0.25),

        # ── Property ──────────────────────────────────────────────────
        "property_type": rng.choice(PROPERTY_TYPE_VARIANTS[prop_type]),
        "bedrooms":      maybe_null(messy_bedrooms(bedrooms), 0.05),
        "bathrooms":     maybe_null(
            str(bathrooms) if rng.random() > 0.12 else f"{bathrooms} bath", 0.07),
        "car_spaces":    maybe_null(
            car_spaces if rng.random() > 0.08 else f"{car_spaces} car", 0.08),
        "toilets":       maybe_null(toilets, 0.20),
        "land_size":     maybe_null(
            messy_land_size(land_sqm) if rng.random() > 0.15 else land_sqm, 0.14),
        "building_area": maybe_null(
            build_sqm if rng.random() > 0.12 else f"{build_sqm:.0f}sqm", 0.35),
        "year_built":    maybe_null(yr_built_val, 0.28),
        "has_pool":      maybe_null(messy_bool(rng.random() < 0.12), 0.18),
        "has_garage":    maybe_null(messy_bool(car_spaces > 0), 0.15),

        # ── Price & sale ──────────────────────────────────────────────
        "sale_price":    messy_price(raw_price),
        "price_raw_aud": maybe_null(
            raw_price if rng.random() > 0.10 else messy_null(), 0.06),
        "sale_date":     maybe_null(messy_date(sale_date), 0.04),
        "sale_method":   maybe_null(method_messy, 0.06),
        "days_on_market": maybe_null(
            dom if rng.random() > 0.06 else rng.choice(["N/A","-","unknown"]), 0.22),
        "inspection_note": inspection_note,

        # ── Agent ──────────────────────────────────────────────────────
        "agent_name":    maybe_null(
            agent if rng.random() > 0.06 else agent.upper(), 0.08),
        "agency_name":   maybe_null(
            agent if rng.random() > 0.08 else agent.lower(), 0.06),
        "agent_phone":   maybe_null(
            f"03 {rng.integers(1000,9999)} {rng.integers(1000,9999)}" if state=="VIC"
            else f"02 {rng.integers(1000,9999)} {rng.integers(1000,9999)}",
            0.12),

        # ── Market context (real conditions for this period) ───────────
        "rba_cash_rate_pct":       mc["rba_rate"],
        "market_sentiment":        mc["sentiment"],
        "market_context":          maybe_null(mc["notes"], 0.15),
        "suburb_median_price":     maybe_null(
            round(base_price * mc["price_index"] / 1000) * 1000, 0.18),
        "auction_clearance_rate_pct": maybe_null(
            round(mc["clearance_mel"] if state=="VIC" else mc["clearance_syd"]
                  + rng.normal(0, 4), 1), 0.30),
        "weekly_rent_aud":         maybe_null(weekly_rent, 0.45),
        "property_count_suburb":   maybe_null(int(rng.integers(200, 15000)), 0.15),
    }

    # Occasional field swaps
    if rng.random() < 0.01:
        row["suburb"], row["council_area"] = row["council_area"], row["suburb"]
    for col in ["bedrooms","bathrooms","car_spaces","days_on_market"]:
        if rng.random() < 0.04 and not pd.isna(row[col]):
            row[col] = messy_null()
    if isinstance(row["postcode"], (int, np.integer)) and rng.random() < 0.08:
        row["postcode"] = f"0{row['postcode']}"

    return row


def inject_duplicates(df, rate=0.02):
    n = max(1, int(len(df) * rate))
    dupes = df.sample(n).copy()
    for col in ["sale_date","agent_phone","source","days_on_market"]:
        if col in dupes.columns:
            dupes[col] = dupes[col].apply(lambda v: messy_null() if rng.random() < 0.4 else v)
    exact = dupes.sample(max(1, n // 3))
    return pd.concat([df, dupes, exact], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# User input helpers
# ─────────────────────────────────────────────────────────────────────────────

MONTH_NAMES = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
    "january":1,"february":2,"march":3,"april":4,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
}

# Data is available from Jan 2020 up to (but not including) the current month.
# VALID_YEARS is computed at runtime so the script never accepts a future date.
from datetime import date as _date
_TODAY         = _date.today()
_DATA_START    = _date(2020, 1, 1)
_DATA_END      = _date(_TODAY.year, _TODAY.month, 1)   # first day of current month (exclusive)

VALID_YEARS  = list(range(_DATA_START.year, _DATA_END.year + 1))
VALID_MONTHS = list(range(1, 13))
VALID_STATES = ["VIC", "NSW", "QLD", "WA", "SA", "TAS", "ACT", "NT"]

# Base listings per month at normal volume (scales with vol_factor)
BASE_ROWS_PER_MONTH = 300


# ─────────────────────────────────────────────────────────────────────────────
# Input parsers
# ─────────────────────────────────────────────────────────────────────────────

def parse_years(raw: str) -> list[int]:
    """
    Two modes:
      Specific  →  2021         single year
                   2021, 2023   comma-separated list
      Interval  →  2021-2023    inclusive range
      Special   →  all          every supported year (2020 – last completed month)

    Validation:
      - Must be a 4-digit number
      - Cannot be before 2020 (data starts Jan 2020)
      - Cannot be in the future beyond the current month
      - Cannot be a non-numeric string
    """
    raw = raw.strip().lower()
    if raw in ("all", "*"):
        return list(VALID_YEARS)

    years = set()
    for part in re.split(r"[,\s]+", raw.strip()):
        part = part.strip()
        if not part:
            continue

        # Reject anything that isn't digits or a dash/en-dash between digits
        if not re.match(r"^[\d\-–]+$", part):
            raise ValueError(
                f"'{part}' is not a valid year — only numbers are accepted.\n"
                "  Specific:  2021   or   2021, 2023\n"
                "  Interval:  2021-2023\n"
                "  All:       all"
            )

        # Interval: 2021-2023
        interval = re.match(r"^(\d{4})\s*[-–]\s*(\d{4})$", part)
        if interval:
            a, b = int(interval.group(1)), int(interval.group(2))
            if a > b:
                raise ValueError(
                    f"Year interval {a}-{b} is reversed — put the earlier year first."
                )
            years |= set(range(a, b + 1))
        elif re.match(r"^\d{4}$", part):
            years.add(int(part))
        else:
            raise ValueError(
                f"'{part}' is not a recognised year format.\n"
                "  Specific:  2021   or   2021, 2023\n"
                "  Interval:  2021-2023\n"
                "  All:       all"
            )

    # Check each year individually for a precise error message
    for y in sorted(years):
        if y < _DATA_START.year:
            raise ValueError(
                f"{y} is before the data start date (January 2020). "
                f"Please choose a year from 2020 onwards."
            )
        if y > _TODAY.year:
            raise ValueError(
                f"{y} is in the future — no data exists yet. "
                f"The latest available year is {_TODAY.year}."
            )

    if not years:
        raise ValueError("No valid years found — please try again.")
    return sorted(years)


def parse_months(raw: str) -> list[int]:
    """
    Two modes:
      Specific  →  3            single month number (1-12)
                   jan          month name
                   1, 3, jun    comma-separated mix of numbers and names
      Interval  →  jan-mar      named range
                   3-6          numeric range
      Special   →  all          every month

    Validation:
      - Month numbers must be 1–12
      - Month names must be recognised (jan, january, feb, etc.)
      - No non-numeric / non-alpha garbage
      - Future-month filtering is done in main() once the year is also known
    """
    raw = raw.strip().lower()
    if raw in ("all", "*"):
        return list(VALID_MONTHS)

    months = set()
    for part in re.split(r"[,\s]+", raw.strip()):
        part = part.strip().rstrip(".")
        if not part:
            continue

        # Reject clearly invalid tokens (not letters, digits, or a dash)
        if not re.match(r"^[a-z\d\-–]+$", part):
            raise ValueError(
                f"'{part}' is not a valid month — use a name (jan) or number (3).\n"
                "  Specific:  jan   or   3   or   jan, mar, jun\n"
                "  Interval:  jan-jun   or   3-6\n"
                "  All:       all"
            )

        # Named interval: jan-mar
        named_interval = re.match(r"^([a-z]+)\s*[-–]\s*([a-z]+)$", part)
        # Numeric interval: 3-6
        num_interval   = re.match(r"^(\d{1,2})\s*[-–]\s*(\d{1,2})$", part)

        if named_interval:
            a = MONTH_NAMES.get(named_interval.group(1))
            b = MONTH_NAMES.get(named_interval.group(2))
            if a is None:
                raise ValueError(
                    f"'{named_interval.group(1)}' is not a recognised month name. "
                    "Use: jan feb mar apr may jun jul aug sep oct nov dec"
                )
            if b is None:
                raise ValueError(
                    f"'{named_interval.group(2)}' is not a recognised month name. "
                    "Use: jan feb mar apr may jun jul aug sep oct nov dec"
                )
            if a > b:
                raise ValueError(
                    f"Month interval '{part}' is reversed — "
                    f"put the earlier month first (e.g. jan-jun, not jun-jan)."
                )
            months |= set(range(a, b + 1))

        elif num_interval:
            a, b = int(num_interval.group(1)), int(num_interval.group(2))
            if a < 1 or b > 12:
                raise ValueError(
                    f"Month numbers must be between 1 and 12 (got {a}-{b})."
                )
            if a > b:
                raise ValueError(
                    f"Month interval {a}-{b} is reversed — "
                    f"put the earlier month first (e.g. 3-9, not 9-3)."
                )
            months |= set(range(a, b + 1))

        elif part in MONTH_NAMES:
            months.add(MONTH_NAMES[part])

        elif re.match(r"^\d{1,2}$", part):
            n = int(part)
            if n < 1 or n > 12:
                raise ValueError(
                    f"{n} is not a valid month number — must be between 1 (January) and 12 (December)."
                )
            months.add(n)

        else:
            raise ValueError(
                f"'{part}' is not a recognised month name or number.\n"
                "  Names:  jan feb mar apr may jun jul aug sep oct nov dec\n"
                "  Numbers: 1–12\n"
                "  Interval: jan-jun  or  3-9\n"
                "  All:      all"
            )

    if not months:
        raise ValueError("No valid months found — please try again.")
    return sorted(months)


def parse_states(raw: str) -> list[str]:
    """
    Two modes:
      Specific  →  VIC           single state
                   VIC, NSW      comma-separated list
      All       →  all           every supported state
    Accepts full names too: Victoria, Queensland, etc.
    """
    NAME_TO_CODE = {
        "victoria": "VIC", "vic": "VIC",
        "new south wales": "NSW", "nsw": "NSW",
        "queensland": "QLD", "qld": "QLD",
        "western australia": "WA", "wa": "WA",
        "south australia": "SA", "sa": "SA",
        "tasmania": "TAS", "tas": "TAS", "tassie": "TAS",
        "australian capital territory": "ACT", "act": "ACT", "canberra": "ACT",
        "northern territory": "NT", "nt": "NT",
    }
    raw = raw.strip().lower()
    if raw in ("all", "*"):
        return VALID_STATES[:]

    states = []
    for part in re.split(r"[,]+", raw):
        part = part.strip()
        if not part:
            continue
        code = NAME_TO_CODE.get(part)
        if code and code not in states:
            states.append(code)
        else:
            # Try uppercase match directly
            upper = part.upper()
            if upper in VALID_STATES and upper not in states:
                states.append(upper)
            elif code is None and upper not in VALID_STATES:
                raise ValueError(
                    f"'{part}' is not a recognised Australian state or territory.\n"
                    "  Accepted: VIC, NSW, QLD, WA, SA, TAS, ACT, NT\n"
                    "  Or full names: Victoria, Queensland, etc.\n"
                    "  Or: all"
                )

    if not states:
        raise ValueError("No valid states found — please try again.")
    return states


def ask(prompt: str, parser, allow_blank: bool = False, blank_default=None):
    """Prompt the user, parse their input, and retry on error."""
    while True:
        raw = input(prompt).strip()
        if allow_blank and raw == "":
            return blank_default
        try:
            return parser(raw)
        except ValueError as e:
            print(f"\n  ✗  {e}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Market preview table
# ─────────────────────────────────────────────────────────────────────────────

def print_market_preview(years: list[int], months: list[int]) -> None:
    periods = [(y, m) for y in years for m in months if (y, m) in MARKET_CONDITIONS]
    if not periods:
        return
    print()
    print("  ┌─────────────┬──────────┬──────────┬──────────┬────────────┬──────────────────────────────────────────────────────┐")
    print("  │ Period      │ RBA rate │ Mel clr% │ Syd clr% │ Sentiment  │ Real-world context                                   │")
    print("  ├─────────────┼──────────┼──────────┼──────────┼────────────┼──────────────────────────────────────────────────────┤")
    for y, m in periods:
        mc    = MARKET_CONDITIONS[(y, m)]
        label = f"{calendar.month_abbr[m]} {y}"
        note  = mc["notes"][:52]
        print(f"  │ {label:<11s} │  {mc['rba_rate']:>5.2f}%  │"
              f"   {mc['clearance_mel']:>3.0f}%   │   {mc['clearance_syd']:>3.0f}%   │"
              f" {mc['sentiment']:<10s} │ {note:<52s} │")
    print("  └─────────────┴──────────┴──────────┴──────────┴────────────┴──────────────────────────────────────────────────────┘")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global rng

    last_avail = _date(_DATA_END.year, _DATA_END.month, 1) - __import__('datetime').timedelta(days=1)
    last_str   = f"{calendar.month_abbr[last_avail.month]} {last_avail.year}"
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║    Australian Housing Market – Messy Data Generator               ║")
    print("║    Prices, clearance rates & sentiment reflect real conditions     ║")
    print(f"║    Data available: January 2020 – {last_str:<30s}  ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    print("  ── Year input ───────────────────────────────────────────────────")
    print("    Specific   →  2021           single year")
    print("                  2021, 2023     individual years")
    print("    Interval   →  2021-2023      every year from 2021 to 2023")
    print("    All        →  all")
    print()
    print("  ── Month input ──────────────────────────────────────────────────")
    print("    Specific   →  jan            single month by name")
    print("                  3              single month by number")
    print("                  jan, mar, oct  individual months, mixed format")
    print("    Interval   →  jan-jun        every month from January to June")
    print("                  3-9            every month from March to September")
    print("    All        →  all")
    print()

    years  = ask("  Which year(s)?   ", parse_years)
    months = ask("  Which month(s)?  ", parse_months)

    # ── Cross-validate: reject any (year, month) that is in the future ───
    # We do this here, not inside the individual parsers, because the month
    # parser doesn't know which year the user chose.
    future = [
        (y, m) for y in years for m in months
        if _date(y, m, 1) >= _DATA_END
    ]
    if future:
        future_strs = [f"{calendar.month_abbr[m]} {y}" for y, m in sorted(future)]
        last_avail_m = _DATA_END.month - 1 or 12
        last_avail_y = _DATA_END.year if _DATA_END.month > 1 else _DATA_END.year - 1
        cutoff_str   = f"{calendar.month_abbr[last_avail_m]} {last_avail_y}"
        print()
        print("  The following month(s) are in the future or not yet available:")
        for s in future_strs:
            print(f"    \u2717  {s}")
        print(f"  Data is available up to {cutoff_str}.")
        print()
        valid_pairs = [(y, m) for y in years for m in months if _date(y, m, 1) < _DATA_END]
        if not valid_pairs:
            print("  No valid periods remain. Please run the script again.")
            sys.exit(1)
        years  = sorted({y for y, m in valid_pairs})
        months = sorted({m for y, m in valid_pairs})
        print("  Continuing with available periods only.")
    print()
    print("  ── State / territory input ──────────────────────────────────────")
    print("    Specific   →  VIC              single state")
    print("                  VIC, NSW, QLD    comma-separated list")
    print("    Full names →  Victoria, Queensland")
    print("    All        →  all              every state and territory")
    print()
    states = ask("  Which state(s)?  ", parse_states)

    # ── Resolve valid (year, month) pairs ────────────────────────────────
    periods = [(y, m) for y in years for m in months if (y, m) in MARKET_CONDITIONS]
    if not periods:
        print("\n  ✗  No valid year/month combinations in the supported range (2020–2025).")
        sys.exit(1)

    # ── Filter suburb pool to selected states ────────────────────────────
    suburb_pool = [s for s in SUBURBS if s[3] in states]
    if not suburb_pool:
        print(f"\n  ✗  No suburbs found for state(s): {states}.")
        sys.exit(1)

    # ── Show real conditions for the chosen period ────────────────────────
    print()
    print(f"  ── Market conditions for your selection ({chr(0x2500) * 44})")
    print_market_preview(years, months)
    print(f"  ── States selected: {', '.join(states)}  ({len(suburb_pool)} suburbs in pool)")

    # Seed from current time so every run produces a fresh, unique dataset.
    import time
    run_seed = int(time.time() * 1000) % (2**31)
    rng = np.random.default_rng(run_seed)
    random.seed(run_seed)

    # ── Row count derived from period, not from user input ────────────────
    # Each month gets BASE_ROWS_PER_MONTH scaled by that month's vol_factor
    # (vol_factor < 1 = quiet market like lockdown, > 1 = busy boom market)
    all_rows = []
    row_id   = 1
    for (y, m) in periods:
        mc           = MARKET_CONDITIONS[(y, m)]
        period_count = max(50, round(BASE_ROWS_PER_MONTH * mc["vol_factor"]))
        for _ in range(period_count):
            all_rows.append(generate_row(row_id, y, m, suburb_pool))
            row_id += 1

    total = len(all_rows)
    print(f"  Generating {total:,} rows across {len(periods)} period(s)")
    print(f"  (~{BASE_ROWS_PER_MONTH} base rows/month, scaled by market volume)")

    df = pd.DataFrame(all_rows)

    print("  Injecting duplicates…")
    df = inject_duplicates(df, rate=0.025)
    df = df.sample(frac=1).reset_index(drop=True)

    # ── Auto-generate filename from selection ─────────────────────────────
    year_part  = (str(min(years)) if len(years) == 1
                  else f"{min(years)}-{max(years)}")
    month_part = (calendar.month_abbr[min(months)].lower() if len(months) == 1
                  else f"{calendar.month_abbr[min(months)].lower()}-"
                       f"{calendar.month_abbr[max(months)].lower()}")
    state_part = ("all_states" if len(states) == len(VALID_STATES)
                  else "-".join(s.lower() for s in sorted(states)))
    out = os.path.join(os.getcwd(), f"aus_housing_messy_{year_part}_{month_part}_{state_part}.csv")

    df.to_csv(out, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print(f"  ✓  Saved → {os.path.abspath(out)}")
    print(f"     {len(df):,} rows   ·   {len(df.columns)} columns")
    print()

    print("  ── Rows generated per period ────────────────────────────────────")
    period_counts = df.groupby(
        df["sale_date"].apply(lambda x: str(x)[:7] if pd.notna(x) else "unknown")
    ).size()
    for y, m in periods:
        mc    = MARKET_CONDITIONS[(y, m)]
        label = f"{calendar.month_abbr[m]} {y}"
        base  = round(BASE_ROWS_PER_MONTH * mc["vol_factor"])
        print(f"     {label:<10s}  vol_factor {mc['vol_factor']:.2f}  →  ~{base} rows  "
              f"(RBA {mc['rba_rate']:.2f}%  ·  sentiment: {mc['sentiment']})")
    print()

    print("  ── Missing value rates ──────────────────────────────────────────")
    miss = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    for col, pct in miss[miss > 0].items():
        bar = "█" * int(pct / 5)
        print(f"     {col:<35s}  {pct:5.1f}%  {bar}")
    print()

    print("  ── Price index vs Jan 2020 base ─────────────────────────────────")
    for y, m in periods:
        mc   = MARKET_CONDITIONS[(y, m)]
        chg  = (mc["price_index"] - 1) * 100
        sign = "+" if chg >= 0 else ""
        print(f"     {calendar.month_abbr[m]} {y}   index {mc['price_index']:.3f}  "
              f"({sign}{chg:.1f}% vs Jan 2020)   RBA {mc['rba_rate']:.2f}%")
    print()

    print("  ── Mess catalogue ───────────────────────────────────────────────")
    print("""
     1.  sale_price              – "$1.25M" / "Contact Agent" / "POA" / "$780,000.00"
     2.  suburb                  – RICHMOND / richmond / richmond_ / " Richmond"
     3.  state                   – 7+ variants: VIC / Vic / victoria / vic. / V.I.C
     4.  sale_date               – 9 date formats mixed throughout
     5.  property_type           – 30+ variants across 7 clean types
     6.  bedrooms                – "3 bed" / "three" / "3.0" / "03" / null strings
     7.  land_size               – sqm int / "650 sqm" / "0.065 ha" / "650m2"
     8.  has_pool / has_garage   – Yes/yes/Y/True/true/1  vs  No/no/N/False/false/0
     9.  nulls                   – NaN, "N/A", "n/a", "-", "unknown", " ", "?"
    10.  postcode                – int vs string; ~6% mismatched to suburb
    11.  year_built              – outlier values: 1066, 9999, 2099
    12.  ~2.5% duplicates        – exact and near-duplicate re-listings
    13.  source whitespace       – "Domain " / " REA"
    14.  rba_cash_rate_pct       – real RBA rate stamped per record
    15.  auction_clearance_rate  – real period clearance ± noise
  """)


if __name__ == "__main__":
    main()