import re
from typing import List
import torch
from transformers import PreTrainedTokenizer
# E-Prime denotes a restricted form of English in which authors avoid all forms of the verb to be.
# E-Prime excludes forms such as be, being, been, present tense forms (am, is, are),
# past tense forms (was, were) along with their negative contractions (isn't, aren't, wasn't, weren't),
# and nonstandard contractions such as ain't and 'twas.
# E-Prime also excludes contractions such as I'm, we're, you're, he's, she's, it's, they're, there's, here's, where's, when's, why's, how's, who's, what's, and that's.
# https://en.wikipedia.org/wiki/E-Prime


BANNED_WORDS = {
    # Base Forms
    "am", "is", "are", "was", "were", "be", "being", "been",
    
    # Negated Forms (Standard & Colloquial)
    "isn't", "aren't", "wasn't", "weren't", "ain't", "amn't", "an't", "hain't",
    "isnt", "arent", "wasnt", "werent", "wasn", "isnt", "aren", "isn", " wasn", # Forms without apostrophes
    
    # Standard Contractions
    "i'm", "you're", "we're", "they're", "he's", "she's", "it's", 
    "that's", "there's", "here's", "who's", "what's", "where's", "when's", 
    "why's", "how's", "who're", "what're", "where're", "when're", "why're", "how're",
    "there're", "they've", "i've", "you've", "we've", "they've", "wouldn", "wouldn't"
    
    # Contractions without apostrophes
    "im", "youre", "were", "theyre", "hes", "shes", "its", 
    "thats", "theres", "heres", "whos", "whats", "wheres", "whens", 
    "whys", "hows", "whore", "whatre", "wherere", "whenre", "whyre", "howre",
    "therere", "theyve", "ive", "youve", "weve", "theyve",
    
    # Possessive vs. Contraction Ambiguities
    "john's", "mary's", "person's", "people's", "man's", "woman's", "child's",
    "men's", "women's", "children's", "company's", "country's", "world's", "day's",
    "year's", "month's", "week's", "hour's", "minute's", "second's", "time's",
    
    # Possessive Pronouns (not E-Prime violations but included for completeness)
    "his", "hers", "its", "theirs", "ours", "yours", "mine",
    
    # Archaic Forms
    "art", "wast", "wert", "beest", "beeth", "'tis", "'twas", "'twere", "thou'rt",
    "doth", "hath", "thou art", "ye be", "thee be", "tis", "twas", "twere",
    
    # Future Forms
    "will be", "shall be", "'ll be", "going to be", "gonna be", "about to be",
    "will have been", "shall have been", "would have been",
    
    # Modal + "Be" Forms
    "must be", "should be", "would be", "could be", "can be", "may be", "might be",
    "ought to be", "has to be", "have to be", "had to be", "supposed to be", 
    "meant to be", "'d be", "used to be", "need be", "dare be",
    
    # Common Passive Voice Constructions (be + past participle)
    "is made", "are made", "was made", "were made", "been made",
    "is done", "are done", "was done", "were done", "been done", 
    "is created", "are created", "was created", "were created", "been created",
    "is seen", "are seen", "was seen", "were seen", "been seen",
    "is known", "are known", "was known", "were known", "been known",
    "is found", "are found", "was found", "were found", "been found",
    "is given", "are given", "was given", "were given", "been given",
    "is called", "are called", "was called", "were called", "been called",
    "is written", "are written", "was written", "were written", "been written",
    "is read", "are read", "was read", "were read", "been read",
    "is spoken", "are spoken", "was spoken", "were spoken", "been spoken",
    "is shown", "are shown", "was shown", "were shown", "been shown",
    "is built", "are built", "was built", "were built", "been built",
    "is considered", "are considered", "was considered", "were considered", "been considered",
    "is required", "are required", "was required", "were required", "been required",
    "is needed", "are needed", "was needed", "were needed", "been needed",
    "is expected", "are expected", "was expected", "were expected", "been expected",
    "is allowed", "are allowed", "was allowed", "were allowed", "been allowed",
    "is permitted", "are permitted", "was permitted", "were permitted", "been permitted",
    "is believed", "are believed", "was believed", "were believed", "been believed",
    "is thought", "are thought", "was thought", "were thought", "been thought",
    "is said", "are said", "was said", "were said", "been said",
    
    # Common Phrases and Idiomatic Expressions
    "so be it", "if need be", "be that as it may", "as it were", "let it be",
    "that is to say", "there are", "there is", "this is", "these are", "those are",
    "it is", "i am", "you are", "we are", "they are", "he is", "she is",
    "be it", "so it is", "that is", "such is", "what is", "this is how", 
    "be sure", "be careful", "be aware", "be ready", "be prepared",
    "let there be", "be my guest", "be yourself", "be honest", "be good",
    
    # Conditional Forms
    "if it is", "if i am", "if you are", "if we are", "if they are",
    "when it is", "when i am", "when you are", "when we are", "when they are",
    "because it is", "because i am", "because you are", "because we are", "because they are",
    "while it is", "while i am", "while you are", "while we are", "while they are",
    "since it is", "since i am", "since you are", "since we are", "since they are",
    "although it is", "although i am", "although you are", "although we are", "although they are",
    "as it is", "as i am", "as you are", "as we are", "as they are",
    
    # Additional Tenses
    "has been", "have been", "had been", 
    "will have been", "would have been", "could have been", "should have been", "might have been",
    "must have been", "may have been", "should've been", "could've been", "would've been",
    
    # Continuous Forms
    "am being", "is being", "are being", "was being", "were being",
    "have been being", "has been being", "had been being",
    "will be being", "would be being", "could be being", "should be being",
    
    # Questions and Interrogative Forms
    "am i", "is he", "is she", "is it", "are we", "are you", "are they",
    "was i", "was he", "was she", "was it", "were we", "were you", "were they",
    "will i be", "will he be", "will she be", "will it be", "will we be", "will you be", "will they be",
    "would i be", "would he be", "would she be", "would it be", "would we be", "would you be", "would they be",
    
    # Interrogative Contractions
    "isn't it", "aren't you", "wasn't he", "weren't they", "haven't been", "hasn't been", "hadn't been",
    
    # Technical Terms (potentially exempt but included for strictness)
    "boolean", "being", "beings", "human being", "living being",
    
    # Common Names/Terms That Contain "Is"
    "isis", "isiah", "ishmael", "isidore", "islington", "islamabad", "isle", "israel", "istanbul",
    
    # Common Stems Requiring Context Analysis
    "beingness", "beingly", "isness", "amness", "areness", "wasness", "wereness",
    
    # Compound Words and Prefixes
    "bespoke", "besmirch", "beset", "bestow", "beside", "besides", "besiege",
    
    # Third-person Present Tense Verbs That End in "is"
    "analysis", "diagnosis", "thesis", "hypothesis", "synthesis", "crisis", "axis", "basis",
    "genesis", "nemesis", "synopsis", "metamorphosis", "psychosis", "neurosis",

    # edge cases
    "seemed", "having", "it's own", "became", "souls", "in a state", "one that", "appeared", "looked", "to have"
}
BANNED_WORDS |= {w.upper() for w in BANNED_WORDS}
BANNED_WORDS |= {w.title() for w in BANNED_WORDS}


class EPrimeLogitsProcessor:
    SPLIT_REGEX = re.compile(r"[^\w']+")

    def __init__(self, tokenizer: PreTrainedTokenizer, max_buffer_chars: int = 50):
        self.tokenizer = tokenizer
        self.max_buffer_chars = max_buffer_chars


        self.candidate_tokens = {}
        eprime_substrings = [w for w in BANNED_WORDS]  # e.g. ["is", "are", "was", ...]
        eprime_substrings_lower = set(eprime_substrings)    # for membership checks

        vocab_size = len(tokenizer)
        for token_id in range(vocab_size):
            token_str = tokenizer.decode([token_id], skip_special_tokens=True)
            if not token_str.strip():
                continue  # empty or purely special/whitespace

            low = token_str.lower()

            if any(sub in low for sub in eprime_substrings_lower):
                self.candidate_tokens[token_id] = token_str

        self.rolling_buffer = ""

    def __call__(
        self,
        prompt_tokens_ids: List[int],
        past_token_ids: List[int],
        scores: torch.Tensor
    ) -> torch.Tensor:
        current_text = self.tokenizer.decode(prompt_tokens_ids + past_token_ids, skip_special_tokens=True)
        self.rolling_buffer = current_text[-self.max_buffer_chars:]

        updated_scores = scores.clone()

        for token_id, token_str in self.candidate_tokens.items():
            hypot_text = (self.rolling_buffer + token_str)
            hypot_text = hypot_text[-(self.max_buffer_chars + len(token_str)):]
            tokens = self.SPLIT_REGEX.split(hypot_text)
            tokens = [t for t in tokens if t]  # remove empty strings
            if tokens:
                last_word = tokens[-1].lower()
                if last_word in BANNED_WORDS:
                    updated_scores[token_id] = float('-inf')

        return updated_scores
