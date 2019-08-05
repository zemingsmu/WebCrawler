import PorterStemmer as ps
import numpy as np
import pandas as pd
import operator


class Engine:

    def __init__(self, K, stop_words, doc_term_list, df, docs):
        self.K = K
        self.query = ''
        self.stop_words = stop_words
        self.term_list = doc_term_list
        self.df = df
        self.docs = docs
        self.thesaurus = {'beautiful': ['nice', 'fanci'], 'chapter': ['chpt'], 'chpt': ['chapter'],
                          'respons': ['owner', 'account'], 'freemanmoor': ['freeman', 'moor'], 'dept': ['depart'],
                          'brown': ['beig', 'tan', 'auburn'], 'tue': ['Tuesdai'],
                          'sole': ['owner', 'singl', 'shoe', 'boot'], 'homework': ['hmwk', 'home', 'work'],
                          'novel': ['book', 'uniqu'], 'comput': ['cse'], 'stori': ['novel', 'book'],
                          'hocuspocu': ['magic', 'abracadabra'], 'thiswork': ['thi', 'work']}

        self.start_engine()

    def content_terms_generate(self, content):
        content_dict = dict()
        # Generate a query term list from the query.
        stemmer = ps.PorterStemmer()
        content_terms = []
        term_list = content.strip().lower().split()
        for word in term_list:
            if word in self.stop_words:
                term_list.remove(word)
        for term in term_list:
            content_terms.append(stemmer.stem(term, 0, len(term)-1))
        # Modify the query term list to a dictionary, then to a vector (Pandas Series).
        for term in content_terms:
            content_dict[term] = 1 if term not in content_dict.keys() else content_dict[term]+1
        return content_dict

    def query_to_vector(self, query_terms):
        query_dict = dict()
        for term in query_terms:
            if term not in query_dict.keys():
                query_dict[term] = 1
            else:
                query_dict[term] += 1
        n = np.zeros(len(self.term_list), dtype=int)
        query_vector = pd.Series(data=n, index=self.term_list)
        for term, frequency in query_dict.items():
            if term in query_vector:
                query_vector[term] = frequency
        return query_vector

    def query_vector_normalize(self, query_vector):
        # Use ntc for query normalization.
        self.df = pd.Series(data=self.df, index=self.term_list).astype(float).as_matrix()
        self.df = np.log10(len(self.docs) / np.abs(self.df))
        df_vector = pd.Series(data=self.df, index=self.term_list)
        result = query_vector.astype(float).multiply(df_vector)
        result = result / np.sqrt((np.sum(np.square(np.array(result)))))
        return result

    def ranking(self, query_vector, query_dict):
        rank_result = dict()
        for d in self.docs:
            d_v = d.get_vector()
            score = d_v.dot(query_vector)
            score = 0 if np.isnan(score) else score
            # Also have to do normalization for title.
            title_dict = self.content_terms_generate(d.get_title())
            for term in query_dict.keys():
                if term in title_dict.keys():
                    score += 0.25
            if score != 0:
                rank_result[d] = score
        rank_score = sorted(rank_result.items(), reverse=True, key=operator.itemgetter(1))
        return rank_score

    def print_result(self, rank_result):
        print('\nThe Query is:', self.query)
        if len(rank_result) == 0:
            print('Sorry, we haven\'t found any result for you, please revise the query.')
        print('Find', len(rank_result), 'results for you. Show the Top', min(len(rank_result), self.K), '!')
        count_1 = 0
        for r in rank_result:
            count_2 = 0
            description = ''
            count_1 += 1
            if count_1 > self.K:
                break
            print('----   ----   ----   ----   ----   ----   ----   ----')
            print('Result No.', count_1, '\nDocument ID:', r[0].get_id(), '\nTitle:', r[0].get_title(), ' \nURL:',
                  r[0].get_url(),  '\nScore:', r[1])
            for char in r[0].get_content():
                if char == ' ':
                    count_2 += 1
                    if count_2 % 10 == 0:
                        description += '\n'
                    if count_2 >= 20:
                        break
                description += char
            print('Page Content:\n', description)

    def query_expansion(self, rank_result, query_dict):
        rank_result = dict(rank_result)
        print('Not enough results, we\'ll use query expansion here.')
        for term in query_dict.keys():
            if term in self.thesaurus.keys():
                for alter in self.thesaurus[term]:
                    query_dict[alter] = query_dict[term]
                    del query_dict[term]
                    query_vector = self.query_to_vector(query_dict)
                    query_vector = self.query_vector_normalize(query_vector)
                    rank_result.update(dict(self.ranking(query_vector, query_dict)))
                    query_dict[term] = query_dict[alter]
                    del query_dict[alter]
        return rank_result

    def start_engine(self):
        while 1:
            self.query = input("Enter your query here!\n")
            if self.query == 'stop':
                break
            # Term dictionary of the query
            query_dict = self.content_terms_generate(self.query)
            # Modify the dictionary to a pandas series
            query_vector = self.query_to_vector(query_dict)
            # Normalize the pandas series
            query_vector = self.query_vector_normalize(query_vector)
            rank_result = self.ranking(query_vector, query_dict)
            if len(rank_result) < self.K/2:
                # If there's no enough results, use thesaurus
                rank_result = sorted(self.query_expansion(rank_result, query_dict).items(),
                                     reverse=True, key=operator.itemgetter(1))
            self.print_result(rank_result)
        print('$$ The Engine is closed~ $$')