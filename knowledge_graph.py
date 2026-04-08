knowledge_graph = {

"telephone": "Alexander Graham Bell",
"gravity": "Isaac Newton",
"penicillin": "Alexander Fleming",
"oxygen": "Antoine Lavoisier",
"electron": "J. J. Thomson",
"radioactivity": "Henri Becquerel",
"x-rays": "Wilhelm Röntgen",
"insulin": "Frederick Banting",
"neptune": "Johann Galle",
"uranus": "William Herschel",
"dna": "James Watson and Francis Crick",
"relativity": "Albert Einstein",
"quantum theory": "Max Planck",
"neutron": "James Chadwick",
"proton": "Ernest Rutherford",
"pluto": "Clyde Tombaugh",
"printing press": "Johannes Gutenberg",
"light bulb": "Thomas Edison",
"telegraph": "Samuel Morse",
"radio": "Guglielmo Marconi",
"television": "John Logie Baird",
"steam engine": "James Watt",
"airplane": "Wright brothers",
"automobile": "Karl Benz",

"hamlet": "William Shakespeare",
"macbeth": "William Shakespeare",
"romeo and juliet": "William Shakespeare",
"pride and prejudice": "Jane Austen",
"war and peace": "Leo Tolstoy",
"odyssey": "Homer",
"iliad": "Homer",

"python": "Guido van Rossum",
"java": "James Gosling",
"c programming language": "Dennis Ritchie",
"c++": "Bjarne Stroustrup",
"linux": "Linus Torvalds",
"world wide web": "Tim Berners-Lee",
"internet": "Vint Cerf",
"facebook": "Mark Zuckerberg",
"amazon": "Jeff Bezos",
"google": "Larry Page and Sergey Brin",
"capital of france": "paris",
"capital of germany": "berlin",
"capital of india": "new delhi",
"capital of japan": "tokyo",
"capital of china": "beijing",

"hobbit": "j r r tolkien",
"harry potter": "j k rowling",
"dracula": "bram stoker",

"ai": "john mccarthy",
"machine learning": "arthur samuel",

"wifi": "john o sullivan",
"bluetooth": "jaap haartsen",

"tesla": "elon musk",
"spacex": "elon musk"
}

import re

def clean(text):
    return re.sub(r'[^a-z0-9 ]', '', text.lower())

def validate_knowledge(question, answer):

    question = clean(question)
    answer = clean(answer)

    if not answer.strip():   
        return 0

    for key in knowledge_graph:

        if key in question:

            correct = clean(knowledge_graph[key])

            # flexible matching
            words = correct.split()
            match_count = sum(1 for w in words if w in answer)

            if correct in answer or match_count >= len(words)//2:
                return 1

    return 0