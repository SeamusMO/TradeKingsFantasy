from player import Player
import csv
import re
from google import genai
import pytesseract
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from PIL.Image import Image as PILImage
from typing import List, Tuple
import os
import re
import numpy as np
import difflib
from dotenv import load_dotenv
import string
from rapidfuzz import fuzz, process
playerNames: list[str] = []
sport = ''
site = ''

load_dotenv()
API_KEY = os.getenv("API_KEY")
client = genai.Client(api_key=API_KEY)
TES_PATH = os.getenv("TESSERACT_CMD")
if TES_PATH:
    pytesseract.pytesseract.tesseract_cmd = TES_PATH
else:
    print("Warning: TESSERACT_CMD environment variable not set. Using default tesseract path.")

SCRIPT_DIR = os.path.dirname(__file__)

def setURL(URL):
    global sport
    global site
    if "nba" in URL or "basketball" in URL:
        sport = 'Basketball'
    elif "nfl" in URL or "football" in URL:
        sport = 'Football'

    if "yahoo" in URL:
        site = 'Yahoo'
    elif "espn" in URL:
        site = 'ESPN'
    elif "sleeper" in URL:
        site = 'Sleeper'

def updatePlayerNames():
    global playerNames
    playerNames = []
    if sport == 'Basketball':
        file_name = 'nbaPlayers.csv'
        expected_cols = 2 # For row[1]
        col_index = 1
    else: # Football
        file_name = 'NFL_player_stats_season_2024.csv'
        expected_cols = 5 # For row[4]
        col_index = 4

    file_path = os.path.join(SCRIPT_DIR, file_name)

    with open(file_path, mode='r', encoding='utf-8') as file:
        csvFile = csv.reader(file)
        for row in csvFile:
            if len(row) > col_index: # Ensure the column exists
                playerNames.append(row[col_index])

def getPlayerNames() -> list[str]:
    global playerNames
    playerNames = []
    if sport == 'Basketball':
        file_name = 'nbaPlayers.csv'
        expected_cols = 2 # For row[1]
        col_index = 1
    else: # Football
        file_name = 'NFL_player_stats_season_2024.csv'
        expected_cols = 5 # For row[4]
        col_index = 4

    file_path = os.path.join(SCRIPT_DIR, file_name)

    with open(file_path, mode='r', encoding='utf-8') as file:
        csvFile = csv.reader(file)
        for row in csvFile:
            if len(row) > col_index: # Ensure the column exists
                playerNames.append(row[col_index])
    return playerNames

def rankings(player):
    if sport == 'Basketball':
        file_name = 'nbaPlayerRankings.csv'
        col_to_match = 1
        col_to_return = 0
    elif sport == 'Football':
        file_name = 'FantasyPros-consensus-rankings.csv'
        col_to_match = 1
        col_to_return = 0
    else:
        return None # Unknown sport

    file_path = os.path.join(SCRIPT_DIR, file_name)
    with open(file_path, mode='r', encoding='utf-8') as file:
        csvFile = csv.reader(file)
        for row in csvFile:
            if len(row) > col_to_match and row[col_to_match] == player:
                if len(row) > col_to_return:
                    return row[col_to_return]
    return None # Player not found or ranking not available

def playerInfo(player):
    if sport == 'Basketball':
        file_name = 'nbaPlayers.csv'
        col_to_match = 1 # Player name is at index 1
        col_to_return = 2 # Position is at index 2
    elif sport == 'Football':
        file_name = 'NFL_player_stats_season_2024.csv'
        col_to_match = 4 # Player name is at index 4
        col_to_return = 5 # Position is at index 5
    else:
        return 'Unknown Position' # Unknown sport

    file_path = os.path.join(SCRIPT_DIR, file_name)
    with open(file_path, mode='r', encoding='utf-8') as file:
        csvFile = csv.reader(file)
        for row in csvFile:
            if len(row) > col_to_match and row[col_to_match] == player:
                if len(row) > col_to_return:
                    return row[col_to_return]
                else:
                    return 'Unknown Position' # Position column missing
    return 'Unknown Position' # Player not found

def fuzzy_match_name(ocr_name: str, threshold: float = 0.75):
    player_list = playerNames
    cleaned = remove_junk(ocr_name)

    # If the line is empty or too short after cleaning, it was junk. Stop here.
    if not cleaned or len(cleaned) < 4:
        return ""

    best_match = ""
    best_score = 0.0

    for known_name in player_list:
        # We use a higher threshold (0.8) to prevent "Kylepittssre" matching "Kyle Trask"
        score = difflib.SequenceMatcher(None, cleaned, known_name.lower().replace(" ", "")).ratio()

        if score > best_score:
            best_score = score
            best_match = known_name

    # Check: If the score is low, don't return anything.
    # This prevents junk like "Ath" from becoming a player.
    if best_score < threshold:
        return ""

    return best_match

def remove_junk(line: str):
    # 1. Patterns for Schedule artifacts (Sunpm, Monspm, Znd, etc.)
    schedule_junk = r'\b(sun|mon|tue|wed|thu|fri|sat)(am|pm|opm|spm|th|sth|nd|rd|ao|oi|ia|itth|ath)\b'

    # 2. Patterns for Position/Team mashups (bufqb, indrb, balwr, phik, sfte, etc.)
    # This looks for 2-3 letters (team) followed by 1-3 letters (position)
    position_junk = r'\b(ari|atl|bal|blt|buf|car|chi|cin|clv|cle|dal|den|det|gb|hst|hou|ind|jax|kc|lv|lac|la|lar|mia|min|ne|no|nop|nyg|nyj|phi|pit|sf|sea|tb|ten|was|wsh)(qb|rb|wr|te|k|dst|d)\b'

    # 3. Small noise artifacts (Zg, Ah, Yp, Ih, Proj)
    noise_junk = r'\b(zg|ah|yp|ih|proj|score|status|opp|fpts|avg|last|roster|starters)\b'

    line = line.lower()

    # Apply regex removals
    line = re.sub(schedule_junk, '', line)
    line = re.sub(position_junk, '', line)
    line = re.sub(noise_junk, '', line)

    # Remove dashes, numbers, and symbols
    line = re.sub(r'[^a-z\s]', '', line)

    # Cleanup whitespace
    line = " ".join(line.split())
    return line

def clean_line(line: str) -> str:
    """
    Cleans OCR text to isolate names. Focuses on removing 2-3 letter
    team/position codes that cause false positives.
    """
    # 1. Strip brackets, pipes, and common OCR noise symbols
    cleaned = re.sub(r'[()\[\]|\\£&!@#$%^{}/~:©_,]', ' ', line)

    # 2. Remove Time/Date/Score patterns (Strict)
    cleaned = re.sub(r'\b\d{1,2}:\d{2}\s*(?:AM|PM)\b', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'[\d\.,\+\-]+\s*%', ' ', cleaned)
    cleaned = re.sub(r'[\d\.]+\s*$', ' ', cleaned)
    cleaned = re.sub(r'\b\d+\b', '', cleaned) # standalone numbers

    # 3. Expanded Blacklist: Added more NFL/NBA specific noise found in your images
    jargon_blacklist = [
    'MOVE', 'TOTALS', 'STATS', 'RESEARCH', 'TOTAL', 'ACTION', 'OPP',
    'STATUS', 'MIN', 'FGM', 'FGA', 'FIM', 'FTA', '3PM', 'REB', 'AST',
    'STL', 'BLK', 'TO', 'TD', 'PTS', 'FPTS', 'BENCH', 'UTIL', 'STARTERS',
    'NOVEMBER', 'SLOT', 'PLAYER', 'WIN', 'LOSS', 'W', 'L', 'I', 'R', 'E',
    'O', 'FE', 'SS', 'AE', 'WE', 'GE', 'ON', 'IR', 'DTD', 'LN', 'AZN', 'PAS',
    'FTS', 'FG', 'FT', 'A', 'B', 'C', 'D', 'N', 'RAE', 'CHA', 'UTAH',
    'WEE', 'POR', 'KE', 'MORAR', 'FL', 'EL', 'FI', 'EE', 'ROST', '+', '-',
    'OF', 'AT', 'AS', 'OE', 'OO', 'EEE', 'FIS', 'SGF', '«12', 'I22', '33S', 'ST', 'pm', 'am'
    'PROJ', 'SCORE', 'OPRK', 'PRK', 'AVG', 'LAST', 'SEA', 'SUN', 'BIST', 'US', 'SRE',
    'EVION', 'WET', 'NO', 'CANS', 'SURLSITHESITS', 'NYJ', 'WITH', 'ND', 'IST', 'TH',
    'QB', 'RB', 'WR', 'TE', 'K', 'DEF', 'DST', 'PK', 'FB', 'DB', 'LB', 'DL', 'CB', 'S', 'DE', 'DT', 'OLB', 'ILB', 'FS', 'SS', 'P', 'SPEC', 'T', 'OL', 'G', 'C', 'OG', 'OT',
    'REC', 'YDS', 'ATT', 'PASS', 'RUSH', 'FLEX', 'BYE', 'WEEK', 'INT', 'FG', 'XP', 'PTS', 'FLEX',
    'BAL', 'BUF', 'CIN', 'CLE', 'DEN', 'HOU', 'IND', 'JAX', 'KC', 'LAC', 'LV', 'MIA', 'NE', 'NYJ', 'PIT', 'TEN', 'WAS', 'OB', 'Dist', 'Ea'
    'ARI', 'ATL', 'CAR', 'CHI', 'DAL', 'DET', 'GB', 'PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'CC', 'LAL', 'BKN', 'CLE', 'OKC', 'MIA', 'MIN', 'DAL', 'LAC','IND', 'HOU', 'TOR', 'PHI', 'CHA', 'BOS', 'NYK', 'MIL',
    'DEN', 'GSW', 'PHX', 'WAS', 'ORL', 'ATL', 'UTA', 'SAS', 'SAC', 'POR', 'NOP', 'MEM', 'DET', 'CHI', 'CLE', 'MIA',
    '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th', '11th', '12th', '13th', '14th', '15th', '16th', '17th', '18th', '19th', '20th', '21st', '22nd', '23rd', '24th', '25th', '26th', '27th', '28th', '29th', '30th', '31st', '32nd',
    ]

    words = cleaned.split()
    name_candidates = []

    for word in words:
        u_word = word.strip(string.punctuation).upper()
        # Only keep words that aren't jargon and aren't single-letter junk (except initials)
        if u_word not in jargon_blacklist:
            if len(word) > 1 or word.upper() in ['A', 'J', 'D', 'K', 'T']: # Allow initials for AJ, DK, TJ
                name_candidates.append(word)

    # Reconstruct name and apply smart capitalization
    temp_name = " ".join(name_candidates).strip()
    return " ".join([w if (len(w) <= 2 and w.isupper()) else w.title() for w in temp_name.split()])

def split_multiple_players(line: str) -> list[str]:
    """Detect and split multiple player names on a single line."""
    KNOWN_PLAYER_NAMES = playerNames

    # If line is empty or too short, return as is
    if not line or len(line) < 3:
        return [line]

    words = line.split()
    if len(words) < 2:
        return [line]

    # Find sequences of words that match known players
    player_sequences = []
    i = 0

    while i < len(words):
        # Try to match 2-word names first, then 1-word names
        matched = False

        # Check for 2-word player names
        if i + 1 < len(words):
            two_word = f"{words[i]} {words[i+1]}"
            if fuzzy_match_name(two_word, threshold=0.75) in KNOWN_PLAYER_NAMES:
                player_sequences.append(two_word)
                i += 2
                matched = True

        # Check for 1-word player names
        if not matched:
            if fuzzy_match_name(words[i], threshold=0.75) in KNOWN_PLAYER_NAMES:
                player_sequences.append(words[i])
                i += 1
                matched = True

        # If no match, skip this word
        if not matched:
            i += 1

    # If we found multiple player matches, return them as separate entries
    if len(player_sequences) > 1:
        return player_sequences

    # Otherwise return the original line
    return [line]

def normalize(text):
    """Removes all spaces, special chars, and lowercases everything."""
    return re.sub(r'[^a-zA-Z]', '', text).lower()

def apiCalling(players):
    list_str = ""
    for x in players:
        # specific logic for formatting the string
        list_str += f"{x.name} {x.rank}\n"

    prompt = "Identify weaknesses in the following NBA fantasy basketball team and suggest improvements.\nBreifly search the web for recent news about each players activity (dont retern this just consider it in your calculations).\n Please seperate ideas using ---, Your response should follow this exact format, Strengths: ..., Weaknesses: ..., Improvement Suggestions," + list_str
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text

def apiCallingTrade(playerst1,playerst2):
    team1_str = "team 1:\n"
    for player in playerst1:
        team1_str += f"{player.name} {player.rank}\n"

    team2_str = "team 2:\n"
    for player in playerst2:
        team2_str += f"{player.name} {player.rank}\n"

    prompt = "Suggest potential trade opportunities for the following NBA fantasy basketball team to improve overall performance. Recommend specific trade targets based on team needs.\nBreifly search the web for recent news about each players activity (dont retern this just consider it in your calculations). The players have rankings next to them.\nPlease seperate ideas using ---, Your response should follow this exact format,Full Trade Offer:[Team 1 offers:[•player 1[•player2[•player3[Team 2 Offers:[•player 1...[, Justification: ...[, Potential Benefits: ....[ ('[[' before next suggestion)The teams are split up by name\n" + team1_str + "\n" + team2_str
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text

def parse_text_to_players(player_names):
    """ Takes a list of strings, creates Player objects, and returns them. """
    playerList = []
    for x in player_names:
        rank = rankings(x)
        position = playerInfo(x)
        player = Player(x, True, position, rank)
        playerList.append(player)
    return playerList

def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    width, height = img.size
    img = img.resize((width * 3, height * 3), Image.Resampling.LANCZOS)
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = ImageEnhance.Contrast(img).enhance(1.5)
    return img

def find_vertical_split(img: Image.Image, margin_pct: float = 0.12) -> int:
    """
    Find the x-coordinate to split two columns (left and right) by computing
    a vertical projection (sum of dark pixels) and finding the lowest valley
    between left and right halves. margin_pct excludes small edges (like UI chrome).
    Returns the x coordinate for the vertical split.
    """
    # Work on a small downsample for speed
    small = img.convert("L").resize((int(img.width / 4), int(img.height / 4)), Image.Resampling.LANCZOS)
    arr = np.array(small)
    # Dark pixels -> lower intensity; invert to measure text presence
    col_sums = (255 - arr).sum(axis=0)
    w = col_sums.shape[0]
    left = int(w * margin_pct)
    right = int(w * (1 - margin_pct))
    # Find minimum column sum (valley) in the central region
    central = col_sums[left:right]
    valley_idx = int(np.argmin(central)) + left
    # Map back to original image coordinates
    split_x = int((valley_idx / w) * img.width)
    return max(10, min(img.width - 10, split_x))

def ocr_text_for_region(img: Image.Image) -> List[str]:
    """
    Run OCR on a PIL image region and return cleaned non-empty lines.
    Uses psm 6 for block / column text.
    """
    processed = preprocess_for_ocr(img)
    # Restricts output to only letters and spaces to avoid 'junk' symbols
    ocr_config = r'--psm 6 -c tessedit_char_whitelist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-"'
    try:
        text = pytesseract.image_to_string(processed, config=ocr_config)
    except Exception as e:
        print("Tesseract error:", e)
        text = ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines

def extract_players_from_side(img_side: Image.Image) -> List[str]:
    """
    Given one side (left or right) PIL image, extract list of player names (strings),
    using the same cleaning + fuzzy matching flow you've used.
    """
    raw_lines = ocr_text_for_region(img_side)
    out_names = []
    for line in raw_lines:
        cleaned = clean_line(line)
        if not cleaned:
            continue
        candidates = split_multiple_players(cleaned)
        for c in candidates:
            c = c.strip()
            if len(c) >= 2:  # allow short names but filter empties
                matched = fuzzy_match_name(c, threshold=0.66)
                # fuzzy_match_name returns original if below threshold; do a heuristic
                if matched and len(matched) > 1:
                    out_names.append(matched)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for n in out_names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique

def perform_ocr_image_split_columns(img: Image.Image) -> Tuple[List[str], List[str]]:
    """
    High-level: take full screenshot PIL image -> return (team1_names, team2_names)
    """
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Find vertical split x coordinate
    split_x = find_vertical_split(img) #seperates the two teams

    # Optionally trim small margins around each side
    margin = max(8, int(img.width * 0.02))
    left_box = (margin, 0, split_x - margin, img.height)
    right_box = (split_x + margin, 0, img.width - margin, img.height)

    left_img = img.crop(left_box)
    right_img = img.crop(right_box)

    # Further optionally remove header area by detecting first heavy text row
    # but we can let OCR and cleaning handle it; if header noise appears, we can
    # filter by blacklisting the known team title words later.

    team1_names = extract_players_from_side(left_img)
    team2_names = extract_players_from_side(right_img)

    return team1_names, team2_names

def postprocess_for_ocr(listOfPlayers) -> list:
    players = []
    for player in listOfPlayers:
        if player.position != 'Unknown Position':
            players.append(player)
    return players

def perform_full_parse_from_image(img: Image.Image) -> dict:
    """
    Returns structured dict with teams and player objects (using parse_text_to_players).
    """
    global playerNames
    playerNames = getPlayerNames()
    t1_names, t2_names = perform_ocr_image_split_columns(img)

    t1_players = parse_text_to_players(t1_names)
    t2_players = parse_text_to_players(t2_names)

    t1_players = postprocess_for_ocr(t1_players)
    t2_players = postprocess_for_ocr(t2_players)

    # Optionally produce trade advice. Use get_trade_advice on each team or combined.
    if len(t1_players) > 0 and len(t2_players) > 0:
        advice1 = apiCallingTrade(t1_players, t2_players)
    elif len(t1_players) > 0:
        advice1 = apiCalling(t1_players)
    else:
        advice1 = "No valid players found"
    # Clean advice text a bit so it looks nicer
    advice1 = (advice1 or "").replace('*', '') # Remove all asterisks
    advice1 = re.sub(r'\s+', ' ', advice1).strip() # Consolidate whitespace
    advice1 = advice1.replace('---', ' \n\n') # Add newlines for separators
    advice1 = advice1.replace('[', '\n') # Add newlines for player separators

    '''for players in t1_players:
        print(players.name)
    for players in t2_players:
        print(players.name)  '''

    return {
        "team1": [{"name": p.name, "position": p.position, "rank": p.rank} for p in t1_players],
        "team2": [{"name": p.name, "position": p.position, "rank": p.rank} for p in t2_players],
        "advice": {
            "Advice": advice1,
        }
    }
