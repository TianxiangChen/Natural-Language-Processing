import math

def BLEU_score(candidate, references, n):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing

	INPUTS:
	sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.


	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""
    candidate_list = candidate.split()
    references_list = []
    for ref in references:
        references_list.append(ref.split())

    sent_len = len(candidate_list)
    ref_len = []

    for ref in references_list:
        ref_len.append(len(ref))
    len_diff = [abs(ref - sent_len) for ref in ref_len]

    ref_closest_len = len(references_list[len_diff.index(min(len_diff))])
    brevity = ref_closest_len / sent_len
    # print(brevity)
    BP = 1 if brevity < 1 else math.exp(1 - brevity)
    # print(BP)
    p_list = []

    # n = len(references_list) if n > len(references_list) else n # avoid n to be too big, thus fail the calculation below
    for i in range(1, n+1):
        ref_set = set()
        for ref in references_list:
            for j in range(len(ref) - i + 1):
                if ' '.join(ref[j:j+i]) not in ref_set:
                    ref_set.add(' '.join(ref[j:j+i]))

        match_count = 0
        for j in range(len(candidate_list) - i + 1):
            if ' '.join(candidate_list[j:j+i]) in ref_set:
                match_count += 1
        # if i == 1:
        #     print('unigram match : ' + str(match_count) + ' , n = ' + str(n) + ' length: ' + str(len(candidate_list) - n + 1))
        # if i == 2:
        #     print('bigram match : ' + str(match_count) + ' , n = ' + str(n) + ' length: ' + str(len(candidate_list) - n + 1))
        p_list.append(match_count / (len(candidate_list) - n + 1))

    multiple_res = 1
    for num in p_list:
        multiple_res *= num

    bleu_score = BP * pow(multiple_res, 1/n)

    return '%0.3f' % bleu_score