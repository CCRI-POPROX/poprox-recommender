import numpy as np

from poprox_concepts.domain import ImpressedSection


def section_browsing_weights(
    sections: list[ImpressedSection],
    alpha: float = 0.10,  # abandon probability
    sigma: float = 0.20,  # skip to next section probability
) -> np.ndarray:
    """
    Compute per-item visit probability weights for section-aware RBP.

    Based on a browsing model where after each item the user can:
    - abandon with probability alpha
    - skip to next section with probability sigma
    - continue within section with probability gamma = 1 - alpha - sigma

    Args:
        sections: list of ImpressedSection objects
        alpha: probability of abandoning reading entirely
        sigma: probability of skipping to next section

    Returns:
        numpy array of visit probabilities, one per article across all sections
    """
    gamma = 1.0 - alpha - sigma
    weights = []
    P_s = 1.0  # probability of visiting first section

    for section in sections:
        section_item_probs = []

        for j in range(len(section.impressions)):
            if j == 0:
                P_v = gamma * P_s
            else:
                P_v = gamma * section_item_probs[-1]
            section_item_probs.append(P_v)
            weights.append(P_v)

        # probability of reaching next section
        P_s_next = sigma * P_s  # skip from header
        for p_v in section_item_probs:
            P_s_next += sigma * p_v  # skip from each item
        P_s_next += gamma * section_item_probs[-1]  # continue past last item

        P_s = P_s_next

    return np.array(weights)
