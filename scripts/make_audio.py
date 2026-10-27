"""Pre-record what the app says, so it speaks Urdu on any phone (many have no Urdu text-to-speech).
Uses Microsoft Edge neural voices via the edge-tts package:  pip install edge-tts && python scripts/make_audio.py"""
import asyncio, os
import edge_tts

PHRASES = {
    "10": ("دس روپے", "Ten rupees"), "20": ("بیس روپے", "Twenty rupees"), "50": ("پچاس روپے", "Fifty rupees"),
    "100": ("سو روپے", "One hundred rupees"), "500": ("پانچ سو روپے", "Five hundred rupees"),
    "1000": ("ایک ہزار روپے", "One thousand rupees"), "5000": ("پانچ ہزار روپے", "Five thousand rupees"),
    "ready": ("کیمرہ تیار ہے۔ نوٹ کیمرے کے سامنے رکھیں", "Camera ready. Hold a note in front of the camera."),
}
VOICES = {"ur": "ur-PK-UzmaNeural", "en": "en-IN-NeerjaNeural"}

async def main():
    for lang, voice in VOICES.items():
        os.makedirs(f"site/audio/{lang}", exist_ok=True)
        for key, (ur, en) in PHRASES.items():
            text = ur if lang == "ur" else en
            await edge_tts.Communicate(text, voice, rate="-5%").save(f"site/audio/{lang}/{key}.mp3")
            print(lang, key, text)

asyncio.run(main())
