import pandas as pd
import ollama
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
prompts = pd.read_csv(DATASET_DIR / "prompts.csv")

print("CSV column names:", prompts.columns.tolist())
print("Total prompts:", len(prompts))
model = "llama3.2"
responses = []
print("\nWEEK 8 - LOCAL LLM INFERENCE")
print("=" * 50)
for index, row in prompts.iterrows():

    prompt = str(row["prompt"])

    print(f"\nPrompt {index + 1}:")
    print(prompt)

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"]

        print("\nModel Response:")
        print(answer)

        responses.append({
            "prompt_id": index + 1,
            "prompt": prompt,
            "response": answer,
            "word_count": len(answer.split()),
            "model": model,
            "status": "Success"
        })

    except Exception as error:

        print("Error:", error)

        responses.append({
            "prompt_id": index + 1,
            "prompt": prompt,
            "response": "",
            "word_count": 0,
            "model": model,
            "status": f"Failed: {error}"
        })
result = pd.DataFrame(responses)
result.to_csv(
    DATASET_DIR / "llm_responses.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nllm_responses.csv saved successfully!")
comparison = result[
    ["prompt_id", "prompt", "word_count", "model", "status"]
]

comparison.to_csv(
    DATASET_DIR / "response_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("response_comparison.csv saved successfully!")
with open(
    DATASET_DIR / "conversation_history.txt",
    "w",
    encoding="utf-8"
) as file:

    for item in responses:

        file.write(f"Prompt ID: {item['prompt_id']}\n")
        file.write(f"Prompt: {item['prompt']}\n")
        file.write(f"Response: {item['response']}\n")
        file.write("-" * 50 + "\n\n")

print("conversation_history.txt saved successfully!")
print("\n" + "=" * 50)
print("PROJECT OUTPUT SUMMARY")
print("=" * 50)

print("Total prompts:", len(prompts))
print(
    "Successful responses:",
    (result["status"] == "Success").sum()
)
print(
    "Failed responses:",
    (result["status"] != "Success").sum()
)

print("\nAll output files created successfully!")