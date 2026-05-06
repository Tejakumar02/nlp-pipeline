import requests
import json

BASE_URL = "http://localhost:8000"

# ── Test 1: Basic NER ────────────────────────────────────────────────────────
def test_ner_basic():
    print("\n" + "="*60)
    print("TEST 1: Basic NER Extraction")
    print("="*60)

    payload = {
        "text": "Elon Musk founded SpaceX in Hawthorne, California. He also leads Tesla, which is headquartered in Austin, Texas.",
        "model": "dslim/bert-base-NER",
        "threshold": 0.85
    }

    response = requests.post(f"{BASE_URL}/api/v1/ner/extract", json=payload)

    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Success! Found {data['entity_count']} entities\n")
        print(f"Model used : {data['model_used']}")
        print(f"Time taken : {data['processing_time_ms']} ms")
        print(f"\nEntities found:")
        for ent in data["entities"]:
            print(f"  [{ent['label']}] {ent['text']:20s}  (confidence: {ent['score']:.2%})")
        print(f"\nEntity Groups:")
        for label, group in data["entity_groups"].items():
            names = [e["text"] for e in group["entities"]]
            print(f"  {label}: {names}")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")


# ── Test 2: Custom text input ────────────────────────────────────────────────
def test_ner_custom(text: str):
    print("\n" + "="*60)
    print("TEST 2: Custom Text NER")
    print("="*60)
    print(f"Input: {text[:100]}...")

    payload = {
        "text": text,
        "model": "dslim/bert-base-NER",
        "threshold": 0.80
    }

    response = requests.post(f"{BASE_URL}/api/v1/ner/extract", json=payload)

    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Found {data['entity_count']} entities in {data['processing_time_ms']} ms\n")
        for ent in data["entities"]:
            print(f"  [{ent['label']}] {ent['text']}")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")


# ── Test 3: NER from a .txt file ─────────────────────────────────────────────
def test_ner_from_file(filepath: str):
    print("\n" + "="*60)
    print(f"TEST 3: NER from file → {filepath}")
    print("="*60)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return

    payload = {
        "text": text,
        "model": "dslim/bert-base-NER",
        "threshold": 0.85
    }

    response = requests.post(f"{BASE_URL}/api/v1/ner/extract", json=payload)

    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Found {data['entity_count']} entities\n")
        print(f"Entity types found: {data['unique_entity_types']}")
        print(f"\nAll entities:")
        for ent in data["entities"]:
            print(f"  [{ent['label']}] {ent['text']:30s}  score: {ent['score']:.2%}")

        # Save results to JSON file
        output_file = filepath.replace(".txt", "_ner_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")


# ── Test 4: Batch NER ─────────────────────────────────────────────────────────
def test_ner_batch():
    print("\n" + "="*60)
    print("TEST 4: Batch NER (multiple texts at once)")
    print("="*60)

    texts = [
        "Apple Inc. CEO Tim Cook announced new products in San Francisco.",
        "Amazon was founded by Jeff Bezos in Seattle, Washington.",
        "The Eiffel Tower is located in Paris, France.",
        "Narendra Modi is the Prime Minister of India.",
        "Microsoft acquired LinkedIn for $26 billion in 2016.",
    ]

    response = requests.post(
        f"{BASE_URL}/api/v1/ner/extract/batch",
        json=texts,
        params={"threshold": 0.85}
    )

    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Processed {data['count']} texts\n")
        for i, result in enumerate(data["results"]):
            if "error" in result:
                print(f"  Text {i+1}: ❌ {result['error']}")
            else:
                entities = [f"[{e['label']}] {e['text']}" for e in result["entities"]]
                print(f"  Text {i+1}: {entities}")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")


# ── Test 5: List available models ────────────────────────────────────────────
def test_list_models():
    print("\n" + "="*60)
    print("TEST 5: Available NER Models")
    print("="*60)

    response = requests.get(f"{BASE_URL}/api/v1/ner/models")
    if response.status_code == 200:
        data = response.json()
        print(f"\nAvailable models:")
        for model in data["models"]:
            desc = data["descriptions"].get(model, "")
            print(f"  • {model}")
            print(f"    {desc}")
    else:
        print(f"❌ Error: {response.text}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check server is running
    try:
        r = requests.get(f"{BASE_URL}/health")
        print(f"✅ Server is running: {r.json()}")
    except requests.ConnectionError:
        print("❌ Server is not running. Start it with: uvicorn app.main:app --reload")
        exit(1)

    # Run tests
    test_list_models()
    test_ner_basic()
    test_ner_batch()

    # ── Uncomment and edit these as needed ────────────────────────────────────

    # Test with your own text:
    #test_ner_custom("Your custom text here with names like John Smith at Google in New York.")
    #test_ner_custom("Teja lives in chennai, Tamil Nadu, India. He works at Infosys and his email is teja@infosys.com. His phone number is +91-9876543210.   He was born on 15th August 1990.He plays cricket and football. He is a software engineer with 5 years of experience in the IT industry. He has a degree in Computer Science from Anna University. He is also interested in machine learning and artificial intelligence.He is zonal level chess player and has won several awards in his school and college days. He is a member of the local chess club and participates in tournaments regularly. He is also a fitness enthusiast and goes to the gym regularly to stay healthy and fit.")
    
    # Test with a .txt file:
    # test_ner_from_file(r"D:\nlp-pipeline\your_document.txt")
    test_ner_from_file(r"D:\nlp-pipeline\wifi.txt")