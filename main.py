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
zone_weight = {'title': 1.5, 'snippet': 1.0}
exclude_filetype = ['jpg', 'jpeg', 'png', 'gif', 'tiff', 'psd', 'pdf', 'eps', 'ai', 'indd', 'raw']


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

    For each word in each zone, 2 zone sub-scores are computed for relevant and non-relevant documents respectively.
    Total score for a word is computed by summing the weighted difference of zone scores.

    Score can be based on: Term-Frequency or Probability
    2 new words with the highest total score are added to the new query

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
    score = {
        'relevant': {'title': 0, 'snippet': 0},
        'non_relevant': {'title': 0, 'snippet': 0}
    }

    # Classify search results into doc, based on user feedbacks
    for i in range(num_result):
        if fbs[i]:
            doc['relevant']['title'] += [parse_text(res[i]['title'])]
            doc['relevant']['snippet'] += [parse_text(res[i]['snippet'])]
        else:
            doc['non_relevant']['title'] += [parse_text(res[i]['title'])]
            doc['non_relevant']['snippet'] += [parse_text(res[i]['snippet'])]

    # Create bag of words for analysis
    bag_of_word = set(original_words)
    for zone in doc['relevant']:
        if zone in ['title', 'snippet']:
            bag_of_word.update([
                word for word in list(itertools.chain.from_iterable(doc['relevant'][zone]))
                if (word.isalnum() and (not word.isnumeric()))
            ])

    word_queue = []
    for word in bag_of_word:
        if word not in stopwords:
            for relevance in ['relevant', 'non_relevant']:
                for zone in ['title', 'snippet']:
                    # Compute term-frequency per document zone
                    tf = np.mean([text.count(word) / len(text) for text in doc[relevance][zone]])

                    # Compute probability per document zone
                    prob = np.mean([(1 if word in text else 0) for text in doc[relevance][zone]])

                    # Choose tf or prob as score for a word
                    score[relevance][zone] = prob

            # Total score of a word is computed as the sum of weighted zone scores.
            # Zone scores is computed as the score difference between relevant documents and non_relevant documents
            total_score = 0
            for zone in ['title', 'snippet']:
                total_score += (score['relevant'][zone] - score['non_relevant'][zone]) * zone_weight[zone]
            heapq.heappush(word_queue, (-total_score, word))

    new_words = []
    while len(new_words) < 2:
        this_score, this_word = heapq.heappop(word_queue)
        if this_word not in original_words:
            new_words.append(this_word)

    new_query = ' '.join(original_words + new_words)
    print('Augmenting by  ' + ' '.join(new_words))

    # === The following codes are for Word Ordering ===
    word_rank_list = []  # list for storing word rank in each relevant doc
    for i in range(num_relevant_result):
        full_doc = doc['relevant']['title'][i] + doc['relevant']['snippet'][i]

        word_avg_pos = []
        for word in original_words + new_words:
            if word.lower() in full_doc:
                # # --- Average Position
                # heapq.heappush(
                #     word_avg_pos,
                #     (np.mean([i for i, term in enumerate(full_doc) if term == word.lower()]), word)
                # )

                # --- First Position
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

    # --- Order keywords based on average rank of each keyword
    word_rank = {word: 0 for word in original_words + new_words}  # dict for storing rank of words in each relevant doc
    for item in word_rank_list:
        for rank, word in enumerate(item):
            word_rank[word] += rank
    new_query = ' '.join([word_tuple[0] for word_tuple in sorted(word_rank.items(), key=lambda wt: wt[1])])

    # # --- Order keywords based on most frequent rank in relevant documents
    # word_rank_list = [' '.join(item) for item in word_rank_list]
    # new_query = max(set(word_rank_list), key=word_rank_list.count)

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
