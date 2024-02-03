import sys
import heapq

from googleapiclient.discovery import build

api_key, engine_id, desired_precision, query = None, None, None, None
stopwords = open('proj1-stop.txt', 'r').read().replace('\n', ' ').split()


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

        query = expand_query(query, results, feedbacks)
