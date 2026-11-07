CLOSED_SYSTEM = "You answer questions about 3GPP 5G NR specifications."

OPEN_SYSTEM = (
    "You answer questions about 3GPP 5G NR specifications. "
    "Use ONLY the numbered context excerpts provided. "
    "Cite the excerpts you used like (S1) or (S2)(S3). "
    "Use abbreviations exactly as written; never expand or define an abbreviation "
    "unless an excerpt does so. "
    "If the excerpts contain relevant but partial information, answer with what they say "
    "and briefly state what is missing, using only terms from the excerpts. "
    "Only if they contain nothing relevant, reply exactly: "
    "'I could not find this in the provided specs.' "
    "Do not use outside knowledge."
)


def build_context(excerpts):
    return "\n\n".join(f"(S{i + 1}) {e['text']}" for i, e in enumerate(excerpts))