import sys
import heapq
import collections
import re

from googleapiclient.discovery import build

api_key, engine_id, desired_precision, query = None, None, None, None
stopwords = open('proj1-stop.txt', 'r').read().replace('\n', ' ').split()

#global list of relevant documents
present_doc_links = set()
present_docs = []

def get_ngrams(n = 2):
    '''
    get_ngrams takes in an array of documents and computes all n-grams that have no stopwords
    :param docs: an array of documents. Each document is a string of words sepearated by spaces
    :param n: size of each gram
    :return: array with the stopword-free n-grams
    '''
    global present_docs
    ngrams = []
    for doc in present_docs:
        doc_words = doc.split()
        doc_ngrams = [doc_words[index: index + n] for index, word in enumerate(doc_words) if index <= len(doc_words) - n]
        ngrams+=doc_ngrams
    ngrams = list(map(lambda y: tuple(y), filter(lambda x: set(x).isdisjoint(stopwords), ngrams)))
    return ngrams 


def get_best_ngrams(ngrams, curr_query, k = 1, alpha = 0.5):
    '''
    get_best_ngrams returns the best ngrams in present_docs based on (1) the probability of any word in the n-gram appearing in bag_of word
    and (2) the probability of the entire n_gram appearing in the docs 
    :param curr_query: a tuple consisting of the terms in the current query. We rule out ngrams that contain a word already present in the query
    :param k: the number of best ngrams to return
    :param alpha: A number between 0 and 1 that mediates between (1) and (2) [see above]. A higher value will emphasize (2).
    :return: k best ngrams
    '''
    distinct_ngrams = set(ngrams)
    bag_of_word = [word for ngram in ngrams for word in ngram]
    unit_scores = [(d_ngram, ngrams.count(d_ngram)) for d_ngram in distinct_ngrams]
    word_scores = [(d_ngram, sum([bag_of_word.count(w) for w in d_ngram])) for d_ngram in distinct_ngrams] 

    final_scores = [(unit_scores[i][0], alpha*unit_scores[i][1] + (1 - alpha)*word_scores[i][1]) for i in range(len(unit_scores))]
    # ngrams that contain a word already in the query are ruled out here
    final_scores = list(filter(lambda y: set(y[0]).isdisjoint(set(curr_query)), final_scores))

    return list(map(lambda x: x[0] ,sorted(final_scores, key=lambda x: x[1], reverse=True )[:k]))



def seb_expand_query(original_query, res, fbs):
    global present_doc_links
    global present_docs

    og_q = tuple(original_query.split())
    
    new_doc_r = []
    for i in range(len(fbs)):
        if res[i]['link'] in present_docs or fbs[i] == 0:
            continue
        present_doc_links.add(res[i]['link'])
        new_doc_r.append(res[i]['title'] + ' ' + res[i]['snippet'])

    # remove non-alphanumeric characters from the docs, make all words lowercase
    new_doc_r = list(map(lambda doc: re.sub('[^0-9a-zA-Z\\s]+', '', doc).lower(), new_doc_r))
    
    # add to our global list of relevant documents
    present_docs += new_doc_r
    
    two_grams = get_ngrams()
    print('========> ', two_grams , '<============')
    expand_query = [og_q] + get_best_ngrams(two_grams, og_q)
    print('++-------OLD Query is ----------++++ ', og_q)
    print('++-------Augmenting by ----------++++ ', get_best_ngrams(two_grams, og_q))
    print('++-------Query is NOW ----------++++ ', ' '.join([word for tup in expand_query for word in tup]))
    return ' '.join([word for tup in expand_query for word in tup])


def run_query(key, cx, q):
    service = build("customsearch", "v1", developerKey=key)
    res = (service.cse().list(q=q, cx=cx).execute())
    return res['items']


def get_relevance_feedback(res):
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
    print('Indexing results ....')
    original_q = original_query.split()
    new_q = []

    doc_r = [(res[i]['title'] + ' ' + res[i]['snippet']) for i in range(len(fbs)) if fbs[i] == 1]
    doc_nr = [(res[i]['title'] + ' ' + res[i]['snippet']) for i in range(len(fbs)) if fbs[i] == 0]

    bag_of_word = set(original_q)

    for doc in doc_r:
        bag_of_word.update([word for word in doc.lower().split() if word.isalnum()])

    word_queue = []
    for word in bag_of_word:
        prob_r = sum([(1 if word in doc else 0) for doc in doc_r]) / len(doc_r)
        prob_nr = sum([(1 if word in doc else 0) for doc in doc_nr]) / len(doc_nr)
        if word not in stopwords:
            heapq.heappush(word_queue, (-(prob_r - prob_nr), word))

    new_word = []
    while len(new_q) < len(original_q) + 2:
        this_score, this_word = heapq.heappop(word_queue)
        if this_word in original_q:
            new_q.append(this_word)
        elif len(new_word) < 2:
            new_q.append(this_word)
            new_word.append(this_word)
    new_query = ' '.join(new_q)
    print('Augmenting by  ' + ' '.join(new_word))
    return new_query


if __name__ == "__main__":
    api_key, engine_id = sys.argv[1], sys.argv[2]
    desired_precision, query = float(sys.argv[3]), sys.argv[4].lower()

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

        # query = expand_query(query, results, feedbacks)
        
        query = seb_expand_query(query, results, feedbacks) 

