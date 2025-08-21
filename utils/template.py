# ======================PARAPHRASE===================
PARAPHRASE_INSTRUCTION_TEMPLATE = """
    ### Instruction:
        - You are an expert language processor. Follow the instructions carefully.
        - Only return the answer in the exact format specified. Do not explain or add anything extra.

    ### Task:
        - Split the following sentences into individual clauses. Rewrite each clause (if needed) follow the Subject + Verb + Object (SVO) structure.
        - Replace all pronouns (e.g., he, she, they, it) with the specific nouns they refer to, based on context.
        - Ensure the rewritten text is coherent, natural, and preserves the original meaning.
"""

PARAPHRASE_QUESTION_TEMPLATE = """
    ### Input sentences are:
"""

PARAPHRASE_OUTPUT_TEMPLATE = """
    ### Output Format:
        - Each input sentence should be on a new line (use '\n' as the separator).
        - Each clause should be seperate by a dot (use '.' as the separator).
"""


# ======================PARSE===================
PARSE_INSTRUCTION_TEMPLATE = """
    ### Instruction:
        - You are an expert language processor. Follow the instructions carefully.
        - Only return the answer in the exact format specified. Do not explain or add anything extra.

    ### Task:
        - Identify the object and the action of each clause
        - Annotate with symbol:
            + 'ac' is an action in the clause
            + 'o' is an object in the clause
            + 'adj' is an adjective in the clause
        - Matching action/object must be beetween start and end token:
            + <ac> action </ac>
            + <o> object </o>
            + <adj> adjective </adj>
        - Ensure the rewritten text is coherent, natural, and preserves the original meaning.
"""

PARSE_QUESTION_TEMPLATE = """
    ### Input captions are:
"""

PARSE_OUTPUT_TEMPLATE = """
    ### Output Format:
        - Each input caption should be on a new line (use '\n' as the separator).
        - Each clause should be seperate by a dot (use '.' as the separator).
        - In each clause, annotate the action and object using prefixes:
            + Example: 'A yellow monkey is eating a yellow banana at the shopping mall' →
              'A <adj> yellow </adj> <o> monkey </o> is <a> eating </a>  <adj> yellow </adj> <o> banana </o> at the <o> shopping mall </o>'
        - Preserve all other parts of the sentence, including subjects and modifiers.
"""


# ======================CAPTIONING===================
CAPTION_INSTRUCTION_TEMPLATE = """
    ### Instruction:
        - You are an expert language processor. Follow the instructions carefully.
        - Only return the answer in the exact format specified. Do not explain or add anything extra.

    ### Task:
        - Carefully analyze the image below and describe everything you can see in the image in as much detail as possible.
        - Categories you should include in the caption are:
            + All Visible objects and people.
            + Describe their actions, positions, and relationships.
            + Mention colors, location, and time of day.
            + Do not overlook small or subtle elements, such as facial expressions, hand gestures, background objects, shadows, or reflections
"""

CAPTION_QUESTION_TEMPLATE = """
    ### Input images are:
"""

CAPTION_OUTPUT_TEMPLATE = """
    ### Output Format:
        - Caption for each input image should be on a new line (use '\n' as the separator).
"""


# ======================CAPTIONING INTERNVL===================
CAPTION_PROMPT = """
    Carefully analyze the image below and describe everything you can see in the image in as much detail as possible. 
    Describe all visible objects and people and their actions, positions, and relationships.
    Mention colors, location, and time of day of the scene in the images.

    ### Captions return in the following format:
        Image 1: <Caption 1>
        ...
        Image n: <Caption n>        
"""


