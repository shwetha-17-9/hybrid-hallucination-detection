from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

documents = [

# Science
"Isaac Newton discovered gravity.",
"Alexander Fleming discovered penicillin in 1928.",
"Antoine Lavoisier discovered oxygen.",
"J.J. Thomson discovered the electron.",
"Henri Becquerel discovered radioactivity.",
"Wilhelm Röntgen discovered X-rays.",
"Frederick Banting discovered insulin.",
"Johann Galle discovered Neptune.",
"William Herschel discovered Uranus.",
"James Watson and Francis Crick discovered the structure of DNA.",
"Albert Einstein developed the theory of relativity.",
"Max Planck developed quantum theory.",
"James Chadwick discovered the neutron.",
"Ernest Rutherford discovered the proton.",
"Clyde Tombaugh discovered Pluto.",
"Galileo Galilei improved the telescope.",
"Zacharias Janssen invented the microscope.",
"Edward Jenner developed the first vaccine.",
"James Clerk Maxwell discovered electromagnetic waves.",
"Charles Darwin developed the theory of evolution.",

# Inventions
"Alexander Graham Bell invented the telephone in 1876.",
"Johannes Gutenberg invented the printing press.",
"Thomas Edison invented the light bulb.",
"Samuel Morse invented the telegraph.",
"Guglielmo Marconi invented the radio.",
"John Logie Baird invented television.",
"James Watt improved the steam engine.",
"Wright brothers invented the airplane.",
"Karl Benz invented the first automobile.",
"Nikola Tesla developed the AC motor.",
"Alfred Nobel invented dynamite.",
"Alexander Bain invented the fax machine.",
"Elisha Otis invented the elevator safety brake.",
"Willis Carrier invented modern air conditioning.",
"Chester Carlson invented the photocopier.",
"László Bíró invented the ballpoint pen.",
"Percy Spencer invented the microwave oven.",
"Douglas Engelbart invented the computer mouse.",
"Robert Noyce invented the integrated circuit.",
"Jack Kilby co-invented the microchip.",

# Geography
"Paris is the capital of France.",
"Berlin is the capital of Germany.",
"Rome is the capital of Italy.",
"Madrid is the capital of Spain.",
"New Delhi is the capital of India.",
"Beijing is the capital of China.",
"Tokyo is the capital of Japan.",
"Seoul is the capital of South Korea.",
"Ottawa is the capital of Canada.",
"Canberra is the capital of Australia.",
"Brasília is the capital of Brazil.",
"Buenos Aires is the capital of Argentina.",
"Moscow is the capital of Russia.",
"Mexico City is the capital of Mexico.",
"Cairo is the capital of Egypt.",
"Pretoria is the administrative capital of South Africa.",
"Ankara is the capital of Turkey.",
"Jakarta is the capital of Indonesia.",
"Bangkok is the capital of Thailand.",
"Hanoi is the capital of Vietnam.",

# Literature
"William Shakespeare wrote Hamlet.",
"William Shakespeare wrote Macbeth.",
"William Shakespeare wrote Romeo and Juliet.",
"Jane Austen wrote Pride and Prejudice.",
"Leo Tolstoy wrote War and Peace.",
"Homer wrote The Odyssey.",
"Homer wrote The Iliad.",
"Dante Alighieri wrote The Divine Comedy.",
"Miguel de Cervantes wrote Don Quixote.",
"F. Scott Fitzgerald wrote The Great Gatsby.",
"Herman Melville wrote Moby Dick.",
"J.D. Salinger wrote The Catcher in the Rye.",
"Harper Lee wrote To Kill a Mockingbird.",
"George Orwell wrote 1984.",
"George Orwell wrote Animal Farm.",
"James Joyce wrote Ulysses.",
"Fyodor Dostoevsky wrote Crime and Punishment.",
"J.R.R. Tolkien wrote The Hobbit.",
"J.R.R. Tolkien wrote The Lord of the Rings.",
"Mark Twain wrote The Adventures of Tom Sawyer.",

# Technology
"Tim Berners-Lee invented the World Wide Web.",
"Vint Cerf is known as a father of the Internet.",
"Guido van Rossum created Python programming language.",
"James Gosling created Java programming language.",
"Dennis Ritchie created the C programming language.",
"Bjarne Stroustrup created C++ programming language.",
"Linus Torvalds created the Linux operating system.",
"Bill Gates co-founded Microsoft.",
"Steve Jobs co-founded Apple.",
"Larry Page and Sergey Brin founded Google.",
"Mark Zuckerberg founded Facebook.",
"Jeff Bezos founded Amazon.",
"Elon Musk founded SpaceX.",
"Satoshi Nakamoto created Bitcoin.",
"Google developed the Android operating system.",
"Apple developed the iOS operating system.",
"Ken Thompson developed the UNIX operating system.",
"IBM developed the first personal computer.",
"Ray Tomlinson invented email.",
"Martin Cooper invented the first mobile phone."

]

model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = model.encode(documents)

dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(doc_embeddings))

def retrieve_context(question):

    query_embedding = model.encode([question])

    D, I = index.search(np.array(query_embedding), k=3)

    results = []

    for idx in I[0]:
        results.append(documents[idx])

    return results

from sentence_transformers import util

def compute_rag_score(answer, context_list, model):
    answer_emb = model.encode(answer, convert_to_tensor=True)
    context_emb = model.encode(context_list, convert_to_tensor=True)

    scores = util.cos_sim(answer_emb, context_emb)

    return float(scores.mean() * 0.7 + scores.max() * 0.3)