"""Validates that a 5-Why chain is actually threaded -- each step should
textually pick up a noun phrase from the previous step, not just be five
loosely-related observations. A keyword-overlap heuristic is enough to catch
the failure mode that matters here: steps that don't build on each other, or a
forward-looking risk statement standing in for a why-answer.

No LLM round-trip: a disconnected chain is truncated to its connected prefix
rather than retried, because a shorter honest chain beats a longer fake one --
consistent with the rest of this pipeline's "never trust, always verify"
philosophy, but here verification-failure just means "stop early," not "ask
again and hope."
"""

from app.services.text_grounding import significant_words


def validate_and_truncate_chain(five_why: list[str]) -> list[str]:
    if len(five_why) <= 1:
        return list(five_why)

    connected = [five_why[0]]
    previous_words = significant_words(five_why[0])

    for step in five_why[1:]:
        step_words = significant_words(step)
        if not (step_words & previous_words):
            # Chain breaks here -- stop rather than pass along a disconnected step.
            break
        connected.append(step)
        previous_words = step_words

    return connected
