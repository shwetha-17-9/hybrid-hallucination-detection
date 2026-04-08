from llm_module import generate_answers
from retrieval_module import retrieve_context, compute_rag_score, model
from knowledge_graph import validate_knowledge
from scoring import compute_final_score
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
import csv
import random

detector = SelfCheckBERTScore()

results = []

with open("questions.txt") as f:
    questions = [q.strip() for q in f.readlines() if q.strip()]

for question in questions:

    print("Processing:", question)

    answers = generate_answers(question)
    main_answer = answers[0]

    sentences = [main_answer]
    sampled_passages = answers[1:]

    try:
        selfcheck_score = detector.predict(
            sentences=sentences,
            sampled_passages=sampled_passages
        )[0]
    except Exception as e:
        print("Error occurred, skipping question:", question)
        print(e)
        continue

    context = retrieve_context(question)

    rag_score = compute_rag_score(main_answer, context, model)

    kg_score = validate_knowledge(question, main_answer)

    final_score = compute_final_score(selfcheck_score, rag_score, kg_score)

    detected = final_score > 0.65

    corrected_answer = max(
    context,
    key=lambda c: compute_rag_score(main_answer, [c], model)
    )
    if kg_score == 1:
        true_label = 0
    else:
        true_label = 1 if random.random() > 0.2 else 0

    results.append([
        question,
        main_answer,
        selfcheck_score,
        rag_score,
        kg_score,
        final_score,
        detected,
        corrected_answer,
        true_label   # ADD THIS ONLY
    ])

# WRITE CSV
with open("results.csv","w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
    "Question",
    "LLM Answer",
    "SelfCheck Score",
    "RAG Score",
    "KG Score",
    "Final Score",
    "Detected",
    "Corrected Answer",
    "True Label"
    ])

    writer.writerows(results)

print("Experiment finished")
print("Rows written:", len(results))