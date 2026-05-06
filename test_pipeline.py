import requests
import json

BASE_URL = "http://localhost:8000"

# Check server
r = requests.get(f"{BASE_URL}/health")
print("Server:", r.json())

# Full pipeline - image → OCR → NER → Summarization
image_path = r"D:\nlp-pipeline\Ima.jpg"  # ← change this to your image path

with open(image_path, "rb") as f:
    response = requests.post(
        f"{BASE_URL}/api/v1/pipeline/process",
        files={"file": ("Ima.jpg", f, "image/jpeg")},
        data={
            "run_ocr": "true",
            "run_ner_stage": "true",
            "run_summarization_stage": "true",
            "ocr_engine": "auto",
            "ner_model": "dslim/bert-base-NER",
            "ner_threshold": "0.85",
            "summarization_model": "facebook/bart-large-cnn",
            "summary_max_length": "150",
            "summary_min_length": "40",
        }
    )

result = response.json()
print(f"\nDocument ID: {result['document_id']}")
print(f"Total time : {result['total_processing_time_ms']} ms")

# OCR
ocr = result.get("ocr")
if ocr and ocr["success"]:
    print(f"\n--- OCR ---")
    print(f"Text: {ocr['data']['text'][:300]}")
    print(f"Confidence: {ocr['data']['confidence']}")

# NER
ner = result.get("ner")
if ner and ner["success"]:
    print(f"\n--- NER ({ner['data']['entity_count']} entities) ---")
    for ent in ner["data"]["entities"]:
        print(f"  [{ent['label']}] {ent['text']}  ({ent['score']:.2%})")

# Summary
summ = result.get("summarization")
if summ and summ["success"]:
    print(f"\n--- Summary ---")
    print(summ["data"]["summary"])
    print(f"Compression: {summ['data']['compression_ratio']:.0%}")

# Save full result
with open("pipeline_result.json", "w") as f:
    json.dump(result, f, indent=2)
print("\n💾 Full result saved to pipeline_result.json")