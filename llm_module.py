from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="gpt2"
)

def generate_answers(question, n=10):

    prompt = f"Question: {question}\nAnswer:"

    answers = []

    for _ in range(n):

        output = generator(
            prompt,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7
        )

        answers.append(output[0]["generated_text"])

    return answers