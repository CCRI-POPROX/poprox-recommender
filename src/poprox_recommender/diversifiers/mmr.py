def mmr_diversification(rewards, similarity_matrix, theta: float, topk: int):
    # MR_i = \theta * reward_i - (1 - \theta)*max_{j \in S} sim(i, j) # S us
    # R is all candidates (not selected yet)

    S = []  # final recommendation (topk index)
    # first recommended item
    S.append(rewards.argmax())

    for k in range(topk - 1):
        candidate = None  # next item
        best_MR = float("-inf")

        for i, reward_i in enumerate(rewards):  # iterate R for next item
            if i in S:
                continue
            max_sim = float("-inf")

            for j in S:
                sim = similarity_matrix[i][j]
                if sim > max_sim:
                    max_sim = sim

            mr_i = theta * reward_i - (1 - theta) * max_sim
            if mr_i > best_MR:
                best_MR = mr_i
                candidate = i

        if candidate is not None:
            S.append(candidate)
    return S
