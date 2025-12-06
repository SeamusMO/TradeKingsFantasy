from player import Player
import csv
import re
from google import genai
import pytesseract
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
from PIL.Image import Image as PILImage
from typing import List, Tuple
import os
import re
import numpy as np
import difflib
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")
client = genai.Client(api_key=API_KEY)
TES_PATH = os.getenv("TESSERACT_CMD")
if TES_PATH:
    pytesseract.pytesseract.tesseract_cmd = TES_PATH
else:
    print("Warning: TESSERACT_CMD environment variable not set. Using default tesseract path.")

def extract_player_names_from_pil(img: PILImage) -> List[str]:
    try:
        # Upscale
        width, height = img.size
        processed_image = img.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
        # Grayscale + enhance
        processed_image = processed_image.convert("L")
        enhancer = ImageEnhance.Contrast(processed_image)
        processed_image = enhancer.enhance(2.0)
        processed_image = processed_image.filter(ImageFilter.SHARPEN)
        # OCR
        ocr_config = r'--psm 6 -l eng'
        text = pytesseract.image_to_string(processed_image, config=ocr_config)
        # --- Post-Processing ---        
        player_names: List[str] = []
        # Split by newline and filter out empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        # Parse OCR output just like extract_player_names()
        players = []
        for line in lines:
            cleaned = clean_line(line)
            possible = split_multiple_players(cleaned)
            for p in possible:
                if len(p) > 3:
                    players.append(fuzzy_match_name(p))

        return players

    except Exception as e:
        print(f"extract_player_names_from_pil error: {e}")
        return []

def getPlayerNames():
    playerNames = []
    with open('nbaPlayers.csv', mode='r', encoding='utf-8') as file:
        csvFile = csv.reader(file)
        for row in csvFile:
            playerNames.append(row[1])
    return playerNames

def rankings(player, list):
    with open('playerRankings.csv', mode='r', encoding='utf-8') as file:
        csvFile = csv.reader(file)
        for row in csvFile:
            if (str(row[1:2]) == ("['" + str(player)) + "']"):
                return row[0]
            
def playerInfo(player, list):
    with open('nbaPlayers.csv', mode='r', encoding='utf-8') as file:
        csvFile = csv.reader(file)
        count = 0
        for row in csvFile:
            if (str(row[1]) == (str(player))):
                return row[2]
        else: 
            return 'Unknown Position'

def fuzzy_match_name(ocr_name: str, threshold: float = 0.7) -> str:
    KNOWN_PLAYER_NAMES = getPlayerNames()
    if not ocr_name:
        return ""

    # Use difflib to find the best match ratio
    best_match = ''
    best_ratio = 0.0
    
    for known_name in KNOWN_PLAYER_NAMES:
        # SequenceMatcher compares two sequences
        ratio = difflib.SequenceMatcher(None, ocr_name.lower(), known_name.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = known_name
            
    # Check if the best match exceeds the required threshold
    if best_ratio >= threshold:
        return best_match
    else:
        # If no good match is found, return the name as is (may still be noise)
        return ocr_name

def clean_line(line: str) -> str:
    # 1. Aggressive symbol and known junk replacement
    
    # 1A. Pre-clean common surrounding symbols that often merge with letters (e.g., '(' ')' '[' ']').
    cleaned = line.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ')
    
    # 1B. Remove Time Patterns: Using strict patterns to avoid truncating names (e.g., 'James').
    # Targets: ' 6:00 PM', ' 7:30 AM', or standalone 'PM'/'AM'.
    cleaned = re.sub(r'\s\d{1,2}:\d{2}\s*[A|P]M', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s([A|P]M)\b', ' ', cleaned, flags=re.IGNORECASE)
    
    # 1C. Remove Stats/Score Patterns: Targets numbers, percentages, and plus/minus symbols.
    cleaned = re.sub(r'[\d\.,\+\-]+\s*%', ' ', cleaned) # Remove percentages, e.g., ' 99.8 %'
    cleaned = re.sub(r'[\d\.]+\s*$', ' ', cleaned)     # Remove trailing numbers/decimals (scores)
    cleaned = re.sub(r'[\+\-][\d\.]+\s*$', ' ', cleaned) # Remove trailing +/- values (like +0.2)
    # Remove game scores, win/loss codes like 'W127-117'
    cleaned = re.sub(r'[W|L]\d{1,3}-\d{1,3}', ' ', cleaned, flags=re.IGNORECASE)
    
    # 1D. Remove long sequences of repeating noise characters that Tesseract often outputs
    cleaned = re.sub(r'[-=]{3,}', ' ', cleaned) # Remove sequences like '---' or '==='
    cleaned = re.sub(r'(?:\s*-\s*){2,}', ' ', cleaned) # Remove spaced dashes like '-- -- --' or ' - -'
    cleaned = re.sub(r'[_]{2,}', ' ', cleaned) # Remove sequences like '__'
    cleaned = re.sub(r'[:]{2,}', ' ', cleaned) # Remove sequences like '::'
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip() # Consolidate multiple spaces
    
    # 1E. Remove common OCR-misread symbols, list separators, and template noise
    cleaned = re.sub(r'[=\\£&!@#$%^{}|/~:©_]', ' ', cleaned) 
    
    # 2. Filter out single words that are common fantasy football jargon, positions, and teams
    
    # Extensive Blacklist for known NBA/Fantasy Football jargon, noise, and short codes
    jargon_blacklist = [
        'MOVE', 'TOTALS', 'STATS', 'RESEARCH', 'TOTAL', 'ACTION', 'OPP', 
        'STATUS', 'MIN', 'FGM', 'FGA', 'FIM', 'FTA', '3PM', 'REB', 'AST', 
        'STL', 'BLK', 'TO', 'TD', 'PTS', 'FPTS', 'BENCH', 'UTIL', 'STARTERS', 
        'NOVEMBER', 'SLOT', 'PLAYER', 'WIN', 'LOSS', 'W', 'L', 'I', 'R', 'E', 
        'O', 'FE', 'SS', 'AE', 'WE', 'GE', 'ON', 'IR', 'DTD', 'LN', 'AZN', 'PAS', 
        'FTS', 'FG', 'FT', 'A', 'B', 'C', 'D', 'N', 'RAE', 'CHA', 'UTAH',
        'WEE', 'POR', 'KE', 'MORAR', 'FL', 'EL', 'FI', 'EE', 'ROST', '+', '-',
        'OF', 'AT', 'AS', 'OE', 'OO', 'EEE', 'FIS', 'SGF', '«12', 'I22', '33S', 'ST',
    ]
    
    # Common Position abbreviations (including single letters)
    position_blacklist = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'CC']
    
    # Common NBA Team abbreviations (Extensive list to filter common noise)
    team_blacklist = ['LAL', 'BKN', 'CLE', 'OKC', 'MIA', 'MIN', 'DAL', 'LAC', 
                      'IND', 'HOU', 'TOR', 'PHI', 'CHA', 'BOS', 'NYK', 'MIL',
                      'DEN', 'GSW', 'PHX', 'WAS', 'ORL', 'ATL', 'UTA', 'SAS',
                      'SAC', 'POR', 'NOP', 'MEM', 'DET', 'CHI', 'CLE', 'MIA']

    # Combine and convert to uppercase for case-insensitive matching
    blacklist = set([w.upper() for w in jargon_blacklist + position_blacklist + team_blacklist])
    
    words = cleaned.split()
    name_candidates = []
    
    for word in words:
        upper_word = word.upper()
        
        # Check if the word is a valid number (integer or float)
        is_num = False
        try:
            float(word)
            is_num = True
        except ValueError:
            pass

        # Check against blacklist and remove standalone numbers
        if upper_word not in blacklist and not is_num:
            # REMOVED AGGRESSIVE FILTERING of 1- and 2-letter uppercase words
            # Relying solely on the comprehensive blacklist to preserve names like 'Ja' and 'Ty'
            name_candidates.append(word)

    # 3. Join words back and clean up extra spaces
    final_name = " ".join(name_candidates).strip()
    
    # 4. Simple capitalization for consistency
    return final_name.title()

def split_multiple_players(line: str) -> list[str]:
    """Detect and split multiple player names on a single line."""
    KNOWN_PLAYER_NAMES = getPlayerNames()
    
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
    
    prompt = "Suggest potential trade opportunities for the following NBA fantasy basketball team to improve overall performance. Recommend specific trade targets based on team needs.\nBreifly search the web for recent news about each players activity (dont retern this just consider it in your calculations)\nPlease seperate ideas using ---, Your response should follow this exact format, Full Trade Offer: ..., Justification: ..., Potential Benefits: .... The teams are split up by name\n" + team1_str + "\n" + team2_str    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text 

def perform_ocr_image(img: Image.Image) -> list[str]:
    return extract_player_names_from_pil(img)

def perform_ocr_bytes(buffer: BytesIO) -> list[str]:
    try:
        buffer.seek(0)
        img = Image.open(buffer)
        return extract_player_names_from_pil(img)
    except Exception as e:
        print(f"perform_ocr_bytes error: {e}")
        return []

def parse_text_to_players(player_names): 
    """ Takes a list of strings, creates Player objects, and returns them. """ 
    playerList = [] 
    for x in player_names: 
        rank = rankings(x, player_names) 
        position = playerInfo(x, player_names) 
        player = Player(x, True, position, rank) 
        playerList.append(player) 
        if rank is None and position is None: 
            playerList.remove(player)
    return playerList

def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """
    Convert to grayscale, upscale, contrast, sharpen, and denoise.
    Returns a processed PIL Image ready for tesseract.
    """
    # Convert to grayscale
    gray = img.convert("L")

    # Upscale 2x using LANCZOS (Pillow 12)
    gray = gray.resize((max(1, gray.width * 2), max(1, gray.height * 2)),
                       Image.Resampling.LANCZOS)

    # Increase contrast
    gray = ImageEnhance.Contrast(gray).enhance(1.9)

    # Slight sharpen
    gray = gray.filter(ImageFilter.SHARPEN)

    # Median filter to reduce small noise (optional)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))

    return gray

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
    ocr_config = r'--psm 6 -l eng'
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
    split_x = find_vertical_split(img)

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

def perform_full_parse_from_image(img: Image.Image) -> dict:
    """
    Returns structured dict with teams and player objects (using parse_text_to_players).
    """
    t1_names, t2_names = perform_ocr_image_split_columns(img)

    t1_players = parse_text_to_players(t1_names)
    t2_players = parse_text_to_players(t2_names)

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
    

    return {
        "team1": [{"name": p.name, "position": p.position, "rank": p.rank} for p in t1_players],
        "team2": [{"name": p.name, "position": p.position, "rank": p.rank} for p in t2_players],
        "advice": {
            "Advice": advice1,
        }
    }
