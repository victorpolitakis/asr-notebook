#!/usr/bin/env python3
"""
generate_bilingual_long.py
Generates a ~12-minute bilingual (Greek + English) podcast WAV file
for pipeline validation. Uses gTTS (Google Text-to-Speech).

Usage:
    pip install gtts pydub
    python generate_bilingual_long.py

Output:
    bilingual_long.wav      (12+ min, 16kHz mono — ready for the pipeline)
    bilingual_long_gt.json  (ground-truth transcript + keywords for evaluation)
"""
import json, os, sys
from pathlib import Path

try:
    from gtts import gTTS
    from pydub import AudioSegment
except ImportError:
    sys.exit("Missing deps: pip install gtts pydub")

# ---------------------------------------------------------------------------
# Script content — alternating English / Greek segments on AI+tech topics
# Persons and orgs are embedded so NER extraction has something to find.
# ---------------------------------------------------------------------------
SEGMENTS = [
    # (lang, text)
    ("en", (
        "Welcome to TechTalks, the bilingual podcast about artificial intelligence "
        "and technology. I'm your host, and today we have a fantastic episode about "
        "how machine learning is transforming the way we learn, communicate, and work. "
        "We'll be talking about models from OpenAI, Google DeepMind, and Anthropic, "
        "and we'll hear from researchers like Geoffrey Hinton, Yann LeCun, and Fei-Fei Li."
    )),
    ("el", (
        "Καλώς ήρθατε στο TechTalks, το δίγλωσσο podcast για την τεχνητή νοημοσύνη και "
        "την τεχνολογία. Σήμερα θα μιλήσουμε για το πώς η μηχανική μάθηση αλλάζει την "
        "εκπαίδευση, την επικοινωνία και την εργασία. Η τεχνητή νοημοσύνη δεν είναι "
        "πλέον επιστημονική φαντασία. Εταιρείες όπως η Google, η Microsoft και η OpenAI "
        "έχουν αναπτύξει μοντέλα που μιλούν, σκέφτονται και δημιουργούν όπως οι άνθρωποι."
    )),
    ("en", (
        "Let's start with the basics. Large Language Models, or LLMs, are neural networks "
        "trained on massive amounts of text data. GPT-4 from OpenAI, Gemini from Google "
        "DeepMind, and Claude from Anthropic are examples of state-of-the-art LLMs. "
        "These models learn statistical patterns in language, allowing them to generate "
        "coherent, contextually relevant text in response to prompts. "
        "Geoffrey Hinton, often called the godfather of deep learning, recently warned "
        "about the existential risks posed by superintelligent AI systems. "
        "Meanwhile, Yann LeCun from Meta argues that current LLMs are fundamentally limited "
        "and cannot achieve human-level reasoning without architectural breakthroughs."
    )),
    ("el", (
        "Τα μεγάλα γλωσσικά μοντέλα, γνωστά ως LLM, είναι νευρωνικά δίκτυα εκπαιδευμένα "
        "σε τεράστιες ποσότητες κειμένου. Το GPT-4 της OpenAI, το Gemini της Google και "
        "το Claude της Anthropic είναι παραδείγματα αιχμής τεχνολογίας. Ο Geoffrey Hinton, "
        "ο λεγόμενος νονός της βαθιάς μάθησης, προειδοποίησε πρόσφατα για τους "
        "υπαρξιακούς κινδύνους που θέτουν τα υπέρ-έξυπνα συστήματα τεχνητής νοημοσύνης. "
        "Η εταιρεία Anthropic, που ιδρύθηκε από πρώην μέλη της OpenAI, εστιάζει στην "
        "ασφαλή ανάπτυξη τεχνητής νοημοσύνης μέσα από τη συνταγματική AI."
    )),
    ("en", (
        "Speech recognition is another area where AI has made remarkable progress. "
        "OpenAI's Whisper model achieves near-human accuracy on multilingual transcription. "
        "Whisper was trained on six hundred thousand hours of multilingual and multitask "
        "supervised data from the web. It supports ninety-nine languages and achieves "
        "a word error rate below five percent on standard benchmarks. "
        "Speaker diarization, which is the task of determining who spoke when, is "
        "handled by models like pyannote audio, developed by Hervé Bredin at LIMSI. "
        "The combination of Whisper and pyannote enables complete podcast processing "
        "pipelines like the one we are demonstrating today."
    )),
    ("el", (
        "Η αναγνώριση ομιλίας είναι ένας τομέας όπου η τεχνητή νοημοσύνη έχει κάνει "
        "εντυπωσιακή πρόοδο. Το μοντέλο Whisper της OpenAI επιτυγχάνει ακρίβεια κοντά "
        "στον άνθρωπο στη μεταγραφή πολλαπλών γλωσσών. Εκπαιδεύτηκε σε εξακόσιες "
        "χιλιάδες ώρες πολύγλωσσων δεδομένων από το διαδίκτυο. Υποστηρίζει "
        "ενενήντα εννέα γλώσσες και επιτυγχάνει ποσοστό λανθασμένων λέξεων κάτω "
        "από πέντε τοις εκατό. Η αρχιτεκτονική βασίζεται σε Transformer encoder-decoder "
        "και χρησιμοποιεί beam search για βέλτιστη αποκωδικοποίηση."
    )),
    ("en", (
        "Natural Language Processing, or NLP, is the branch of AI that deals with "
        "understanding and generating human language. Named Entity Recognition, "
        "or NER, is a fundamental NLP task that involves identifying and classifying "
        "named entities such as persons, organizations, and locations in text. "
        "Modern NER systems use transformer architectures like BERT from Google "
        "and RoBERTa from Facebook AI Research. These models achieve F1 scores "
        "above ninety percent on standard benchmarks like CoNLL-2003. "
        "Summarization is another key NLP task. Extractive summarization selects "
        "key sentences from the source document, while abstractive summarization "
        "generates new text that captures the main ideas. ROUGE metrics, developed "
        "by Chin-Yew Lin, are used to evaluate summarization quality."
    )),
    ("el", (
        "Η επεξεργασία φυσικής γλώσσας, γνωστή ως NLP, είναι ο κλάδος της τεχνητής "
        "νοημοσύνης που ασχολείται με την κατανόηση και παραγωγή ανθρώπινης γλώσσας. "
        "Η αναγνώριση ονοματικών οντοτήτων, ή NER, είναι μια θεμελιώδης εργασία NLP "
        "που περιλαμβάνει την αναγνώριση προσώπων, οργανισμών και τοποθεσιών. "
        "Σύγχρονα συστήματα NER χρησιμοποιούν αρχιτεκτονικές Transformer όπως το "
        "BERT της Google και το RoBERTa του Facebook AI Research. "
        "Η περίληψη είναι μια άλλη βασική εργασία NLP. Τα μετρικά ROUGE, που "
        "αναπτύχθηκαν από τον Chin-Yew Lin, χρησιμοποιούνται για την αξιολόγηση "
        "της ποιότητας των περιλήψεων. Το ROUGE-1 μετρά την επικάλυψη unigram "
        "μεταξύ της παραγόμενης και της αναφοράς περίληψης."
    )),
    ("en", (
        "Let's talk about the computational requirements of modern AI systems. "
        "Training GPT-4 required thousands of NVIDIA A100 GPUs and cost an estimated "
        "one hundred million dollars. Inference, however, is much more affordable. "
        "On Apple Silicon, the Metal Performance Shaders framework enables efficient "
        "neural network inference on the GPU. The M-series chips from Apple "
        "combine CPU, GPU, and Neural Engine on a single die, achieving impressive "
        "performance per watt ratios. For podcast processing pipelines, Apple's "
        "Metal framework reduces transcription latency by a factor of three to five "
        "compared to CPU-only processing. Our pipeline uses CTranslate2 with int8 "
        "quantization, which further reduces memory usage and speeds up inference."
    )),
    ("el", (
        "Ας μιλήσουμε για τις υπολογιστικές απαιτήσεις των σύγχρονων συστημάτων "
        "τεχνητής νοημοσύνης. Η εκπαίδευση του GPT-4 απαίτησε χιλιάδες GPU NVIDIA "
        "A100 και κόστισε εκατό εκατομμύρια δολάρια. Το συμπέρασμα, όμως, είναι "
        "πολύ πιο οικονομικό. Στο Apple Silicon, το Metal Performance Shaders "
        "επιτρέπει αποτελεσματική εκτέλεση νευρωνικών δικτύων στο GPU. "
        "Τα chips M-series της Apple συνδυάζουν CPU, GPU και Neural Engine σε ένα "
        "chip, επιτυγχάνοντας εντυπωσιακές αναλογίες απόδοσης ανά watt. "
        "Το pipeline μας χρησιμοποιεί CTranslate2 με ποσοτικοποίηση int8, "
        "που μειώνει τη χρήση μνήμης και επιταχύνει την εκτέλεση."
    )),
    ("en", (
        "The future of AI in education looks very promising. Personalized learning "
        "systems powered by LLMs can adapt to each student's pace and learning style. "
        "Carnegie Mellon University and MIT have developed AI tutoring systems that "
        "provide immediate feedback and guidance. Khan Academy, in partnership with "
        "OpenAI, has launched Khanmigo, an AI tutor powered by GPT-4. "
        "Duolingo uses AI to personalize language learning for five hundred million "
        "users worldwide. The platform adapts difficulty, pacing, and content based "
        "on individual performance data. This kind of adaptive learning is only "
        "possible because of advances in reinforcement learning from human feedback, "
        "a technique pioneered by researchers at DeepMind and OpenAI."
    )),
    ("el", (
        "Το μέλλον της τεχνητής νοημοσύνης στην εκπαίδευση φαίνεται πολλά υποσχόμενο. "
        "Εξατομικευμένα συστήματα μάθησης που τροφοδοτούνται από LLM μπορούν να "
        "προσαρμοστούν στον ρυθμό και τον τρόπο μάθησης κάθε φοιτητή. "
        "Το Carnegie Mellon University και το MIT έχουν αναπτύξει συστήματα "
        "AI εκπαίδευσης που παρέχουν άμεση ανατροφοδότηση. Το Khan Academy, "
        "σε συνεργασία με την OpenAI, έχει λανσάρει το Khanmigo, έναν AI δάσκαλο "
        "που τροφοδοτείται από το GPT-4. Το Duolingo χρησιμοποιεί τεχνητή νοημοσύνη "
        "για να εξατομικεύσει την εκμάθηση γλωσσών για πεντακόσια εκατομμύρια χρήστες "
        "παγκοσμίως. Αυτό το είδος προσαρμοστικής μάθησης είναι δυνατό μόνο χάρη "
        "στις εξελίξεις στην ενισχυτική μάθηση από ανθρώπινη ανατροφοδότηση."
    )),
    ("en", (
        "Let's wrap up today's episode with a look at AI safety and ethics. "
        "Responsible AI development requires careful attention to bias, fairness, "
        "privacy, and transparency. The European Union's AI Act, passed in 2024, "
        "establishes a risk-based regulatory framework for AI systems. "
        "High-risk applications in healthcare, education, and critical infrastructure "
        "must meet strict requirements for accuracy, robustness, and human oversight. "
        "Sam Altman, CEO of OpenAI, has testified before the United States Senate "
        "about the need for AI regulation. Demis Hassabis, co-founder of DeepMind, "
        "advocates for international cooperation on AI safety research. "
        "The alignment problem — ensuring that AI systems pursue goals that are "
        "beneficial to humanity — remains one of the central challenges of our time."
    )),
    ("el", (
        "Ας ολοκληρώσουμε το σημερινό επεισόδιο με μια ματιά στην ασφάλεια και "
        "την ηθική της τεχνητής νοημοσύνης. Η υπεύθυνη ανάπτυξη AI απαιτεί "
        "προσεκτική προσοχή στην αμεροληψία, την ιδιωτικότητα και τη διαφάνεια. "
        "Ο Νόμος για την Τεχνητή Νοημοσύνη της Ευρωπαϊκής Ένωσης, που εγκρίθηκε "
        "το 2024, θεσπίζει ένα κανονιστικό πλαίσιο βασισμένο στον κίνδυνο. "
        "Ο Sam Altman, διευθύνων σύμβουλος της OpenAI, κατέθεσε ενώπιον της "
        "Αμερικανικής Γερουσίας για την ανάγκη ρύθμισης της τεχνητής νοημοσύνης. "
        "Ο Demis Hassabis, συνιδρυτής της DeepMind, υποστηρίζει τη διεθνή συνεργασία "
        "για την έρευνα ασφάλειας AI. Το πρόβλημα ευθυγράμμισης, που αφορά τη "
        "διασφάλιση ότι τα συστήματα AI επιδιώκουν στόχους ωφέλιμους για την "
        "ανθρωπότητα, παραμένει μία από τις κεντρικές προκλήσεις της εποχής μας."
    )),
    ("en", (
        "Thank you for listening to TechTalks. In today's episode, we covered "
        "large language models from OpenAI, Google DeepMind, and Anthropic. "
        "We discussed speech recognition with Whisper, named entity recognition, "
        "and abstractive summarization. We explored AI's impact on education through "
        "examples from Carnegie Mellon, MIT, Khan Academy, and Duolingo. "
        "Finally, we examined AI safety challenges and regulatory frameworks. "
        "If you enjoyed this episode, please subscribe and share with your friends. "
        "In our next episode, we will dive into computer vision and multimodal AI. "
        "Until next time, keep learning and keep building!"
    )),
]

# ---------------------------------------------------------------------------
# Ground-truth construction
# ---------------------------------------------------------------------------
FULL_TRANSCRIPT = " ".join(text for _, text in SEGMENTS)

GROUND_TRUTH = {
    "transcript": FULL_TRANSCRIPT,
    "summary": (
        "TechTalks podcast covers the state of artificial intelligence and machine learning, "
        "exploring large language models from OpenAI, Google DeepMind, and Anthropic, "
        "including GPT-4, Gemini, and Claude. The episode discusses speech recognition "
        "with OpenAI's Whisper model, NLP tasks including NER and summarization using "
        "ROUGE metrics, and computational requirements on Apple Silicon using Metal "
        "Performance Shaders. AI's transformative role in education is highlighted through "
        "Khan Academy's Khanmigo and Duolingo's personalized learning. The episode concludes "
        "with AI safety and ethics, covering the EU AI Act and alignment challenges "
        "discussed by Geoffrey Hinton, Sam Altman, and Demis Hassabis."
    ),
    "keywords": [
        "artificial intelligence", "machine learning", "OpenAI", "Google DeepMind",
        "Anthropic", "GPT-4", "Gemini", "Claude", "Whisper", "speech recognition",
        "NLP", "NER", "ROUGE", "Apple Silicon", "Metal Performance Shaders",
        "transformer", "BERT", "Khan Academy", "Duolingo", "AI safety",
        "EU AI Act", "alignment", "deep learning", "LLM"
    ],
    "persons": [
        "Geoffrey Hinton", "Yann LeCun", "Fei-Fei Li", "Chin-Yew Lin",
        "Sam Altman", "Demis Hassabis", "Hervé Bredin"
    ],
    "organizations": [
        "OpenAI", "Google DeepMind", "Anthropic", "Meta", "Apple",
        "Carnegie Mellon University", "MIT", "Khan Academy", "Duolingo",
        "NVIDIA", "European Union", "Facebook AI Research"
    ],
    "languages": ["en", "el"],
    "duration_sec_approx": 720,
}

# ---------------------------------------------------------------------------
# Audio generation
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent
WAV_PATH   = OUTPUT_DIR / "bilingual_long.wav"
GT_PATH    = OUTPUT_DIR / "bilingual_long_gt.json"

def generate():
    print(f"Generating {len(SEGMENTS)} TTS segments …")
    combined = AudioSegment.silent(duration=0)
    pause    = AudioSegment.silent(duration=800)   # 0.8 s between segments

    for i, (lang, text) in enumerate(SEGMENTS, 1):
        print(f"  [{i:02d}/{len(SEGMENTS)}] {lang.upper()} — {text[:60]}…")
        tmp = f"/tmp/_seg_{i}.mp3"
        gTTS(text, lang=lang, slow=False).save(tmp)
        seg = AudioSegment.from_mp3(tmp)
        combined += seg + pause
        os.remove(tmp)

    # Normalise to 16 kHz mono (Whisper/Silero standard)
    combined = combined.set_frame_rate(16_000).set_channels(1)
    combined.export(str(WAV_PATH), format="wav")
    duration = len(combined) / 1000
    print(f"\n✅  Audio written → {WAV_PATH}  ({duration:.1f}s / {duration/60:.1f} min)")

    GT_PATH.write_text(json.dumps(GROUND_TRUTH, ensure_ascii=False, indent=2))
    print(f"✅  Ground truth → {GT_PATH}")
    return duration

if __name__ == "__main__":
    dur = generate()
    if dur < 600:
        print(f"⚠️  Audio is only {dur:.0f}s — add more segments for 10+ min.")
    else:
        print(f"🎙️  Ready for pipeline: python run_pipeline.py sample_podcasts/bilingual_long.wav")
