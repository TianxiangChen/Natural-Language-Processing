import re

def preprocess(in_sentence, language):
    """
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation

	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence

	OUTPUT:
	out_sentence: (string) the modified sentence
    """
    out_sentence = ''
    out_sentence = in_sentence.replace('\n','')
    out_sentence = out_sentence.strip()
    out_sentence = procBothLang(out_sentence)

    if language == 'f':
        out_sentence = proFrenchOnly(out_sentence)

    out_sentence = 'SENTSTART ' + out_sentence + ' SENTEND'
    out_sentence = ' '.join(out_sentence.split())
    return out_sentence


def procBothLang(in_sentence):
    """
       This function performs processing on all in_sentence for both languages.
       The processing includes:
        - separate sentence-final punctuation, commas, colons and semicolons,
          parentheses, dashes between parentheses, mathematical operators (e.g., +,-,<,>,=),
          and quotation marks
        - convert all tokens to lower-case

       INPUTS:
       in_sentence : (string) the original sentence to be processed

       OUTPUT:
       out_sentence: (string) the modified sentence
    """
    out_sentence = ''
    out_sentence = in_sentence.lower()


    if out_sentence[-1] in ['.', '!', '?']:
        out_sentence = out_sentence[:-1] + ' ' + out_sentence[-1]
    punc_set = set(['(', ')', '+', '-', '<', '>', '=', '"', ',', ':', ';'])

    for punc in punc_set:
        out_sentence = out_sentence.replace(punc, ' ' + punc + ' ')

    # I didn't explicitly handle the dash case, since it is the same as minus sign in ascii
    out_sentence = re.sub('\s{2,}', ' ', out_sentence)
    return out_sentence


def proFrenchOnly(in_sentence):
    """
       This function performs processing on in_sentence in french.
       The processing includes adding spacing for:
        - Singular definite article(le, la)
        - Single-consonant words ending in e-'muet' (e.g., 'dropped'-e ce. je)
        - que
        - Conjunctions puisque and lorsgue

       INPUTS:
       in_sentence : (string) the sentence to be processed

       OUTPUT:
       out_sentence: (string) the modified sentence
    """
    out_sentence = ''

    out_sentence = re.sub(r"(l'|t'|j'|qu')([a-zA-A])", r"\1 \2", in_sentence)
    out_sentence = re.sub(r"([a-zA-Z]')(on|il)\s", r"\1 \2 ", out_sentence)
    out_sentence = re.sub('\s{2,}', ' ', out_sentence)
    return out_sentence
