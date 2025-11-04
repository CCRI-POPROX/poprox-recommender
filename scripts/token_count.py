from transformers import AutoTokenizer

text = """"The natural or physical world,\
        and especially the relationship between nature \
        (ecosystems, wildlife, the atmosphere, water, land, etc.) \
        and human beings. Includes the effects of human activities \
        on the environment and vice versa, as well as the \
        management of nature by humans. May also include \
        discussions of the natural world that are unrelated \
        to humans or human activity."""

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize(text)

breakpoint()
