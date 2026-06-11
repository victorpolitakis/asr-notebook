import os
from gtts import gTTS
from pydub import AudioSegment

def generate_bilingual_audio():
    print("Synthesizing English part 1...")
    tts1 = gTTS("Hello everyone, welcome to the ECE 22073 advanced audio project. Today we are going to showcase our end-to-end podcast processing pipeline.", lang='en')
    tts1.save("temp1.mp3")

    print("Synthesizing Greek part 2...")
    tts2 = gTTS("Αυτό το σύστημα χρησιμοποιεί υπερσύγχρονα μοντέλα τεχνητής νοημοσύνης για να κάνει αποδελτίωση, ταυτοποίηση ομιλητών, θεματική εξαγωγή και σύνοψη σε πραγματικό χρόνο.", lang='el')
    tts2.save("temp2.mp3")

    print("Synthesizing English part 3...")
    tts3 = gTTS("So, let's trigger the Whisper model zero-shot transcription and evaluate our processing latency.", lang='en')
    tts3.save("temp3.mp3")

    print("Synthesizing Greek part 4...")
    tts4 = gTTS("Καλή επιτυχία σε όλους τους φοιτητές της σχολής!", lang='el')
    tts4.save("temp4.mp3")

    print("Concatenating and converting to 16kHz mono WAV...")
    sound1 = AudioSegment.from_mp3("temp1.mp3")
    sound2 = AudioSegment.from_mp3("temp2.mp3")
    sound3 = AudioSegment.from_mp3("temp3.mp3")
    sound4 = AudioSegment.from_mp3("temp4.mp3")

    # Combine with slight pauses between segments
    pause = AudioSegment.silent(duration=1000)
    combined = sound1 + pause + sound2 + pause + sound3 + pause + sound4

    # Convert to 16kHz mono as expected by the pipeline
    combined = combined.set_frame_rate(16000).set_channels(1)
    
    # Save to the sample_podcasts folder
    os.makedirs("Samples/sample_podcasts", exist_ok=True)
    combined.export("Samples/sample_podcasts/bilingual_test.wav", format="wav")
    print("Exported successfully to Samples/sample_podcasts/bilingual_test.wav!")

    # Cleanup temporary files
    for f in ["temp1.mp3", "temp2.mp3", "temp3.mp3", "temp4.mp3"]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    generate_bilingual_audio()
