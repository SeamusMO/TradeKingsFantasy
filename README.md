# Fantasy Trade Advisor

A Chrome extension and Python-based backend system that processes fantasy basketball roster screenshots, extracts player information using OCR, and provides analytical recommendations to improve team composition and trade strategy.

This project integrates browser automation, image processing, optical character recognition, fuzzy name matching, and AI-driven analysis into a cohesive tool designed to streamline roster evaluation.

---

## Overview

Key capabilities include:

* Automated screenshot capture through a Chrome extension
* Local OCR processing with Tesseract to extract player and roster data
* Fuzzy name matching to extract all player names correct
* AI-generated recommendations using Google GenAI
* A modular architecture separating the browser interface from backend logic
* Secure configuration and secret management through environment variables

---

## Project Structure

```
fantasy-trade-advisor/
│
├── extension/
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   ├── icon.png
|   ├── content.js
|   ├── styles.css
|   ├── background.js
│
├── backend/
│   ├── app.py                   
│   ├── main.py
│   ├── player.py              
│   ├── playerRankings.csv
│   ├── nbaPlayers.csv      
│   ├── requirements.txt
│   ├── config_example.env
│
├── .gitignore
└── README.md
```

---

## Prerequisites

### Python Dependencies

Install all required packages from the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Tesseract OCR

This project requires a local installation of Tesseract OCR.
Downloads and installation instructions are available at:
[https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)

Record the installation path, as it must be provided in the environment configuration.

---

## Environment Configuration

Create a `.env` file inside the `backend/` directory with the following fields:

```
API_KEY=YOUR_GOOGLE_GENAI_KEY
TESSERACT_CMD=PATH_TO_TESSERACT_EXECUTABLE
```

This file should not be committed to version control.
A `config_example.env` file is provided to indicate the required fields.

---

## Running the Backend Server

Start the Flask server from the `backend/` directory:

```bash
python app.py
```

The service will be available at:

```
http://127.0.0.1:5000
```

---

## Loading the Chrome Extension

1. Open Google Chrome and navigate to:
   `chrome://extensions/`
2. Enable **Developer Mode**.
3. Select **Load unpacked**.
4. Choose the `extension/` directory from this repository.

The extension will then appear in the browser’s extension toolbar.

---

## System Workflow

### 1. Screenshot Capture

The Chrome extension captures the visible portion of the fantasy roster webpage through the Chrome Extensions API.

### 2. Image Transmission

The screenshot is encoded (Base64) and transmitted to the backend via an HTTP POST request.

### 3. Image Preprocessing

The backend refines the image using the Pillow library to improve OCR accuracy.
Adjustments include brightness correction, contrast enhancement, noise reduction, and filtering.

### 4. Optical Character Recognition

Tesseract OCR processes the preprocessed image and extracts textual information such as player names, positions, and the team they player for.

### 5. Fuzzy Name Matching

The backend adjusts the outputed names to match the players real names exactly

### 6. AI-Driven Analysis

The extracted data is evaluated using Google GenAI to generate:

* Roster optimization recommendations
* Trade or waiver suggestions (Depending on if you upload one or two teams)
* Positional assessments and comparative insights

### 7. Response Delivery

The backend returns the analysis in a structured JSON format, which the extension displays in the user interface.

---

## License

This project is released under the MIT License.

---

## Acknowledgments

This system utilizes:

* Tesseract OCR
* Google GenAI
* Pillow
* Flask
* Chrome Extensions API

Their respective documentation and communities provided essential support in constructing this project.
