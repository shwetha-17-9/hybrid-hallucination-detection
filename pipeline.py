from llm_module import generate_answers
from retrieval_module import retrieve_context, compute_rag_score, model
from knowledge_graph import validate_knowledge
from scoring import compute_final_score
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore

question = "Who invented the telephone?"

print("\nUser Question:")
print(question)

answers = generate_answers(question)

print("\nGenerated Answers:")

for ans in answers:
    print("-", ans)

detector = SelfCheckBERTScore()

sentences = [answers[0]]
sampled_passages = answers[1:]

selfcheck_score = detector.predict(
    sentences=sentences,
    sampled_passages=sampled_passages
)[0]

context = retrieve_context(question)

rag_score = compute_rag_score(answers[0], context, model)

kg_score = validate_knowledge(question, answers[0])

final_score = compute_final_score(selfcheck_score, rag_score, kg_score)

print("\nFinal Hallucination Score:", final_score)

if final_score > 0.75:

    print("\n⚠ Hallucination detected")

    print("\nRetrieved Knowledge:")

    for c in context:
        print("-", c)

    print("\nCorrected Answer:", context[0])

else:

    print("\nAnswer appears reliable")