import sys
import heapq
import requests
import numpy as np
import re
import itertools
from bs4 import BeautifulSoup
from googleapiclient.discovery import build

api_key, engine_id, desired_precision, query = None, None, None, None
stopwords = open('proj1-stop.txt', 'r').read().replace('\n', ' ').split()
zone_weight = {'title': 1.2, 'snippet': 1.0}
exclude_filetype = ['jpg', 'jpeg', 'png', 'gif', 'tiff', 'psd', 'pdf', 'eps', 'ai', 'indd', 'raw']
bigram_count = {}   # a dictionary storing number of occurrences of each bi-gram


def get_bigram_score(word, alpha=0.8):
    """
    Compute bigram score of a word based on cumulative bigram counts.
    :param word: word to be searched in the list of previous bi-grams
    :param alpha: param to control how much a bi-gram occurrence contributes to the score; range is [0.1, 2]
    :return:
    """

    # Use logarithm to scale number of bi-gram occurrences, with floor set to 1
    base_score = np.log10(sum([value for key, value in bigram_count.items() if word in key]) + 1) + 1
    return base_score ** (max(0.1, min(2, alpha)))


def get_bigram_list(text, allow_stopwords=False):
    """
    Takes in a list of words and generate a list of bi-grams
    :param text: list of words for finding bi-grams
    :param allow_stopwords: if True, stopwords are allowed in bi-grams
    :return: array with the stopword-free bi-grams
    """
    bigrams = [text[index: index + 2] for index, word in enumerate(text) if index <= len(text) - 2]
    bigrams = list(map(
        lambda y: tuple(y), bigrams if allow_stopwords else filter(lambda x: set(x).isdisjoint(stopwords), bigrams)
    ))
    return bigrams


def update_bigram_count(text):
    """
    Take in a list of words and update bigram_counts, which count number of occurrences of each bi-gram cumulatively
    :param text: list of words for finding bi-grams
    """
    global bigram_count

    bigrams = get_bigram_list(text, allow_stopwords=True)
    for bigram in bigrams:
        bigram_count[bigram] = (bigram_count[bigram] + 1) if (bigram in bigram_count) else 1


def add_begin_end_symbol(text):
    """
    Add begin symbol and end symbol to the word list
    :param text: a list of words
    :return: List of begin symbol, words and end symbol
    """
    return ['<s>'] + text + ['</s>']


def parse_text(text):
    """
    Convert text into a lower-case words, removing special characters and space.

    :param text: input text in string
    :return: list of lower-case words
    """
    return [word for word in re.split(r"[^a-zA-Z0-9]", text.lower()) if word]


def run_query(key, cx, q):
    """
    Call Google API and receive list of results.
    Image files or non-HTML files are filtered out

    :param key: Google API key
    :param cx: Google Engine id
    :param q: query in string
    :return: list of filtered results
    """
    service = build("customsearch", "v1", developerKey=key)
    res = (service.cse().list(q=q, cx=cx).execute())

    filtered_res = []
    for r in res['items']:
        if (r['link'].split('.')[-1] in exclude_filetype) or ('fileFormat' in r):
            continue
        filtered_res += [r]
    return filtered_res


def get_relevance_feedback(res):
    """
    Interact with user to get relevance feedback by showing search results
    :param res: filtered search result from Google API
    :return: list of boolean user feedbacks corresponding to the input search results
    """

    user_feedbacks = []

    print('Google Search Results:')
    print('======================')
    for i, res in enumerate(res):

        print('Result' + str(i + 1))
        print('[')
        print('URL: ' + res['link'])
        print('Title: ' + res['title'])
        print('Summary: ' + res['snippet'])
        print(']')
        print()
        feedback = input('Relevant (Y/N)?').lower()
        if feedback not in ['y', 'n']:
            raise Exception('Invalid Input')
        user_feedbacks += [1 if (feedback == 'y') else 0]
    return user_feedbacks


def expand_query(original_query, res, fbs):
    """
    === Word Selection ===
    Search results are considered as document with different zones: Title and Snippet
    Each document is classified as "relevant" or "non-relevant" based on user feedbacks.


    For each word in each zone, 2 zone-sub-scores are computed for relevant and non-relevant documents respectively.
    Total zone-score for a word is computed by summing the weighted difference of zone-scores.
    Zone-score can be based on: Term-Frequency or Probability

    Bi-gram score is computed for each word by calling get_bigram_score().
    It is a function of number of occurrences of bi-grams involving that word.

    Final score is computed as a product of Total zone score and Bi-gram score.
    2 new words with the highest final score are added to the new query

    === Word Ordering ===
    Only relevant documents are used to decide word order in new query.
    For each word in each doc, position is first examined in one of the followings: Average Position or First Position

    Word rank (per doc) for each word is computed according to their relative positions.
    Next query words order is based on one of the followings: Average Rank per word or Most-frequent order

    :param original_query: the query (string) used in last search
    :param res: list of filtered results from last search
    :param fbs: list of user feedbacks corresponding to the filtered results
    :return: a new query string for next search
    """

    print('Indexing results ....')
    original_words = original_query.split()
    num_result = len(res)
    num_relevant_result = sum(fbs)

    # === The following codes are for Word Selection ===
    # Initialize "doc" for storing list of words in each zone, for relevant and non-relevant docs
    doc = {
        'relevant': {'title': [], 'snippet': []},
        'non_relevant': {'title': [], 'snippet': []}
    }

    # Initialize score of a word in each zone, for relevant and non-relevant docs
    zone_score = {
        'relevant': {'title': 0, 'snippet': 0},
        'non_relevant': {'title': 0, 'snippet': 0}
    }

    # Classify search results into doc, based on user feedbacks
    for i in range(num_result):
        if fbs[i]:
            doc['relevant']['title'] += [parse_text(res[i]['title'])]
            doc['relevant']['snippet'] += [parse_text(res[i]['snippet'])]

            # Update bigrams count based on relevant documents
            update_bigram_count(add_begin_end_symbol(doc['relevant']['title'][-1]))
            update_bigram_count(add_begin_end_symbol(doc['relevant']['snippet'][-1]))
        else:
            doc['non_relevant']['title'] += [parse_text(res[i]['title'])]
            doc['non_relevant']['snippet'] += [parse_text(res[i]['snippet'])]

    # Create bag of words for analysis
    bag_of_word = set(original_words)
    for zone in doc['relevant']:
        bag_of_word.update([
            word for word in list(itertools.chain.from_iterable(doc['relevant'][zone]))
            if (word.isalnum() and (not word.isnumeric()))
        ])

    word_queue = []
    for word in bag_of_word:
        if word not in stopwords:
            for relevance in ['relevant', 'non_relevant']:
                for zone in ['title', 'snippet']:
                    # 1. Compute average term-frequency per document zone
                    tf = np.mean([text.count(word) / len(text) for text in doc[relevance][zone]])

                    # 2. Compute probability per document zone
                    prob = np.mean([(1 if word in text else 0) for text in doc[relevance][zone]])

                    # Choose tf or prob as score for a word
                    zone_score[relevance][zone] = prob

            # Final score of a word is computed as the sum of weighted zone scores, adjusted by bigram score
            # Zone scores is computed as the score difference between relevant documents and non_relevant documents
            bigram_score = get_bigram_score(word)
            total_zone_score = sum([
                (zone_score['relevant'][zone] * bigram_score - zone_score['non_relevant'][zone]) * zone_weight[zone]
                for zone in ['title', 'snippet']
            ])
            final_score = total_zone_score

            heapq.heappush(word_queue, (-final_score, word))

    new_words = []
    while len(new_words) < 2:
        this_score, this_word = heapq.heappop(word_queue)
        if this_word not in original_words:
            new_words.append(this_word)
    print('Augmenting by  ' + ' '.join(new_words))
    new_word_set = original_words + new_words

    # === The following codes are for Word Ordering ===
    word_rank_list = []  # list for storing word rank in each relevant doc
    corpus = []
    for i in range(num_relevant_result):
        full_doc = add_begin_end_symbol(doc['relevant']['title'][i]) + \
                   add_begin_end_symbol(doc['relevant']['snippet'][i])
        corpus += [word for word in full_doc if word in (
                add_begin_end_symbol([new_word.lower() for new_word in new_word_set])
        )]

        word_avg_pos = []
        for word in new_word_set:
            if word.lower() in full_doc:
                # # --- 1. Average Position
                # heapq.heappush(
                #     word_avg_pos,
                #     (np.mean([i for i, term in enumerate(full_doc) if term == word.lower()]), word)
                # )

                # --- 2. First Position
                heapq.heappush(
                    word_avg_pos,
                    (full_doc.index(word.lower()), word)
                )
            else:
                heapq.heappush(
                    word_avg_pos,
                    (np.inf, word)
                )
        word_avg_pos = [heapq.heappop(word_avg_pos)[1] for i in range(len(word_avg_pos))]
        word_rank_list.append(word_avg_pos)

    # # --- 1. Order keywords based on average rank of each keyword
    # word_rank = {word: 0 for word in new_word_set}  # dict for storing rank of words in each relevant doc
    # for item in word_rank_list:
    #     for rank, word in enumerate(item):
    #         word_rank[word] += rank
    # new_query = ' '.join([word_tuple[0] for word_tuple in sorted(word_rank.items(), key=lambda wt: wt[1])])

    # # --- 2. Order keywords based on most frequent rank in relevant documents
    # word_rank_list = [' '.join(item) for item in word_rank_list]
    # new_query = max(set(word_rank_list), key=word_rank_list.count)

    # --- 3. Order keywords based on probability from bigrams in corpus
    word_order_bigrams = (get_bigram_list(corpus, allow_stopwords=True))
    word_orders = list(itertools.permutations(new_word_set))
    word_order_log_score = {}
    for word_order in word_orders:
        this_word_order_log_score = 0
        this_word_order = ['<s>'] + [word.lower() for word in list(word_order)] + ['</s>']
        for i in range(len(this_word_order) - 1):
            num_bigram_first_word = len(list(filter(lambda x: x[0] == this_word_order[i], word_order_bigrams)))
            if num_bigram_first_word > 0:
                num_bigram = word_order_bigrams.count((this_word_order[i], this_word_order[i + 1]))
                with np.errstate(divide='ignore'):
                    this_word_order_log_score += np.log(num_bigram / num_bigram_first_word)
        word_order_log_score[word_order] = this_word_order_log_score

    top_word_order = list(sorted(word_order_log_score.items(), key=lambda wt: wt[1], reverse=True)[0][0])
    new_query = ' '.join(top_word_order)

    return new_query


if __name__ == "__main__":
    api_key, engine_id = sys.argv[1], sys.argv[2]
    desired_precision, query = float(sys.argv[3]), sys.argv[4]

    while True:

        print('Parameters:')
        print('Client Key  = ' + api_key)
        print('Engine Key  = ' + engine_id)
        print('Query       = ' + query)
        print('Precision   = ' + str(desired_precision))

        results = run_query(api_key, engine_id, query)
        feedbacks = get_relevance_feedback(results)
        precision = sum(feedbacks) / len(feedbacks)

        print('======================')
        print('FEEDBACK SUMMARY')
        print('Query ' + query)
        print('Precision ' + str(precision))

        if precision >= desired_precision:
            print('Desired precision reached, done')
            break
        else:
            print('Still below the desired precision of ' + str(desired_precision))
            print('Indexing results ....')

        query = expand_query(query, results, feedbacks)
