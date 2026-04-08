def compute_final_score(selfcheck_score, rag_score, kg_score):
    alpha, beta, gamma = 0.2, 0.45, 0.35

    final_score = (
        alpha * (1 - selfcheck_score) +
        beta * (1 - rag_score) +
        gamma * (1 - kg_score)
    )

    return final_score