
# This project is based on Beautiful Soup Parser and PorterStemmer Algorithm
# [1].Porter Stemming Algorithm/Python  Reference from https://tartarus.org/martin/PorterStemmer/python.txt
# [2].Beautiful Soup Documentation      Reference from https://www.crummy.com/software/BeautifulSoup/bs4/doc/
# By Jiaqing Ji

from bs4 import BeautifulSoup
import urllib
from urllib import request
from urllib import error
from urllib import parse
from urllib import robotparser
from queue import Queue as Queue
from Document import Document
from Engine import Engine
import re
from stop_words import get_stop_words
import pandas as pd
import numpy as np
import operator
import time
import random


class Crawlers:
    def __init__(self, url, crawler_name, limit, stop_words, scheme, netloc):
        self.url = url
        self.crawler_Name = crawler_name    # Name of the crawler.
        self.limit = limit                  # Limit of page amount
        self.stop_words = stop_words
        self.url_queue = Queue()            # The URL Frontier queue.
        self.url_visited = set()            # The visited URLs.
        self.url_out = set()                # Out-going links.
        self.url_outof_type = set()         # Page type is out of type scope (like email address).
        self.url_broken = set()             # The broken URLs.
        self.url_disallowed = set()         # Disallowed URLs.
        self.url_image = set()              # Image type of URLs.
        self.url_other = set()              # Other type of URLs.
        self.url_already_seen = set()       # URLs already seem.
        self.url_all = set()                # All appeared URLs.

        self.doc_id = 0                     # Use in file names when write web files into local files.
        self.docs = []                      # A list of documents, each has a unique id.
        self.term_list = []                 # Term list
        self.term_doc = dict()              # A dictionary with term-document information.
        self.matrix = pd.DataFrame()        # Term Document Matrix
        self.leader_matrix = pd.DataFrame() # new-Leader-Follower score matrix.

        # Time information
        self.prev_time = time.time()        # Last time grab an URL from queue.
        self.cur_time = 0                   # Time grab an URL from queue.

        self.scheme = scheme
        self.netloc = netloc
        self.directory = '/~fmoore'

        self.leader_docs = []               # new-
        self.leader_follower = dict()       # new-Dictionary of leader follower information.

    def robots(self, test_url):
        root_re = re.compile(self.directory)
        if root_re.match(test_url):
            test_url = test_url.split(self.directory)[1]
            # print(test_url)
        else:
            test_url = '/' + test_url

        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(self.url + '/robots.txt')
        rp.read()
        if rp.can_fetch(self.crawler_Name, test_url):
            return True
        return False


    def url_formalize(self, current_url, raw_url):
        # Change all urls to absolute path
        # Choose urls that don't go out of the website
        components = parse.urlparse(raw_url)
        url_format = raw_url
        if not self.robots(raw_url):
            # print('Page Not Allowed to Access')
            return ''
        if components.scheme in self.scheme:
            if components.netloc not in self.netloc:
                self.url_out = self.url_out.union({raw_url})
                # print('         Page Out of Root')
                return ''
            else:
                mark = re.compile('/~fmoore')
                if mark.match(components.path) is None:
                    self.url_out = self.url_out.union({raw_url})
                    # print('         Page Out of Root')
                    return ''
        elif components.scheme == '':
            tag = False
            url_format = parse.urljoin(current_url, raw_url)
            for str_a in self.scheme:
                for str_b in self.netloc:
                    mark = re.compile(str_a + '://' + str_b + '/~fmoore')
                    if mark.match(url_format):
                        tag = True
            if not tag:
                url_format = ''
        else:  # Other type of links like emails.
            self.url_outof_type = self.url_outof_type.union({raw_url})
            self.url_out = self.url_out.union({raw_url})
            # print('         Page Type is Out of Root')
            url_format = ''

        return url_format

    def jaccard(self, doc1, doc2):
        # Get the jaccard score of two documents.
        term1 = set(doc1.get_term().keys())
        term2 = set(doc2.get_term().keys())
        jaccard_score = len(term1.intersection(term2))/len(term1.union(term2))
        # We can print jaccard score of the new document with all other documents in docs[] if we want
        # print("Jaccard score of Doc#{} and Doc#{} is: {}".format(doc1.get_id(), doc2.get_id(), jaccard_score))
        return jaccard_score

    def duplicate_detection(self, doc1, doc2):
        return doc1.get_content() == doc2.get_content()

    def parse(self, url, file_type, file_content):
        # Parse the file as a HTML file.
        # Reference from: https://stackoverflow.com/questions
        #   30565404/remove-all-style-scripts-and-html-tags-from-an-html-page
        text = file_content
        title = ''
        if 'html' in file_type:
            # Clean the file. Don't save HTML markup
            soup = BeautifulSoup(file_content, 'html.parser')
            # Remove all javascript and stylesheet code.
            for script in soup(["script", "style"]):
                script.extract()

            title = soup.title.string          # Get the title of this file.
            # print("The title of this file is: ", title)
            text = soup.body.get_text()        # Get the body of this file.

        lines = (line.strip() for line in text.splitlines())
        # Build a chunk of tokens.
        chunks = []
        for line in lines:
            for phrase in line.split(" "):      # Split with space.
                chunks.append(phrase.strip())
        # Drop blank lines.
        text = '\n'.join(chunk for chunk in chunks if chunk)

        # Write to a file.
        self.doc_id += 1
        filename = "Doc#" + str(self.doc_id) + '.txt'
        # Ensure the file will closed.
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)

        # I only give id to document I'm gonna parse.
        document = Document(url, self.doc_id, filename, file_type, self.stop_words)
        document.filter()
        document.stem()
        document.collection()
        # print("There're", len(document.term), "terms in document", filename)

        if 'html' in file_type:
            document.set_title(title)

        # Duplicate Detection
        for d in self.docs:
            if self.duplicate_detection(d, document) == 1:
                # print("The content of Doc#{} is exact duplicate with Doc#{}, so, we won't parse Doc#{}."
                #       .format(document.get_id(), d.get_id(), document.get_id()))
                self.url_already_seen = self.url_already_seen.union({str(document.get_url())})
                return False
        self.docs.append(document)
        return True

    def crawl(self):
        # Processing the url frontier queue.
        self.url_queue.put(self.url)        # Put the root URL in queue.
        self.url_all = self.url_all.union({self.url})
        count = 1

        while self.url_queue.qsize():                           # While the frontier queue is not empty.
            url = self.url_queue.get()
            if url in self.url_visited:
                continue                                        # Ignore this visited one and deal with next URL in queue.

            self.cur_time = time.time()
            # print("previous time: ", self.prev_time)
            # print("current time: ", self.cur_time)
            if (self.cur_time - self.prev_time) < 0.1:
                time.sleep(0.1)
            self.prev_time = time.time()

            req = urllib.request.Request(url, data=None)
            req.add_header('User-Agent', 'Chrome/65.0.3325.162')
            try:
                response = urllib.request.urlopen(req)
            except urllib.error.HTTPError as e:
                # print('\n------------------------------------------------------')
                # print("HTTPError when access " + url)
                # print("         Reason:", e.reason)
                # print('-----------------------------------------------------\n')
                self.url_broken = self.url_broken.union({url})
                continue
            except urllib.error.URLError as e:
                # print('\n------------------------------------------------------')
                # print("URLError when access " + url)
                # print("         Reason:", e.reason)
                # print('-----------------------------------------------------\n')
                self.url_broken = self.url_broken.union({url})
                continue

            self.url_visited = self.url_visited.union({url})

            # print('\n------------------------------------------------------')
            # print('*********        Start with a new file       *********')
            # print('----   ----   ----   ----   ----   ----   ----   -----')

            # Dealing with different types of the response file.
            file_type = response.getheader('Content-Type')
            new_url = response.geturl()
            # print(new_url, 'File_type of this file is:', file_type)
            if 'text/html' in file_type or 'text/plain' in file_type:
                # Get a text/html or text/plain type file.
                # print('This is a text/html document: {}'.format(new_url))
                try:
                    # Ignore the UnicodeError raised when decode.
                    file_content = response.read().decode('utf-8', errors='ignore')
                except e:
                    print("         Reason:", e.reason)
                    continue

                valid = self.parse(new_url, file_type, file_content)
                if not valid:
                    continue

                # Crawl all URLs under this page.
                urlformat_href = re.compile('href=\s*"?(.+?)"?[ >]')
                urls_href = urlformat_href.findall(file_content)
                urlformat_src = re.compile('src=\s*"?(.+?)"?[ >]')
                urls_src = urlformat_src.findall(file_content)
                urls = urls_href + urls_src
                for u in urls:
                    self.url_all = self.url_all.union({u})
                    # print(" Crawl a URL in this page:{}".format(u))
                    url_format = self.url_formalize(new_url, u)
                    if url_format != '':
                        if url_format in self.url_visited:
                            print(' ')
                            # print('     This Page is already Visited.')
                        else:
                            self.url_queue.put(url_format)
                            # print('     Add', url_format, 'to the Queue.')
                # Limit of page amount
                if valid:
                    count += 1
                    if count > self.limit:
                        # print('----   ----   ----   ----   ----   ----   ----   ----')
                        # print('************     Finish this file        ************')
                        # print('-----------------------------------------------------\n')
                        break

            elif 'image' in file_type:      # Get a image file.
                # print(' This is a image document: {}'.format(new_url))
                self.url_image = self.url_image.union({new_url})
            else:
                # print(' This is a special document: {}'.format(new_url))
                self.url_other = self.url_other.union({new_url})
            # print('----   ----   ----   ----   ----   ----   ----   ----')
            # print('************     Finish this file        ************')
            # print('-----------------------------------------------------\n')

    def td_matrix(self):
        doc_list = []
        for d in self.docs:
            doc_list.append('Doc#'+str(d.get_id()))
            for t in d.get_term():
                if t not in self.term_list:
                    self.term_list.append(t)
                if t not in self.term_doc.keys():
                    l = {d}
                    self.term_doc[t] = l
                else:
                    self.term_doc[t] = self.term_doc[t].union({d})
        # Build a term-document matrix
        td_matrix = []
        for t in self.term_doc.keys():
            l = []
            count = 0
            for i in range(len(self.docs)):
                l.append(0)
            for d in self.docs:
                if d in self.term_doc[t]:
                    l[count] = d.get_term()[t]
                count += 1
            td_matrix.append(l)
        # matrix is a pandas data frame
        # Write to a excel file.
        self.matrix = pd.DataFrame(data=td_matrix, index=self.term_list, columns=doc_list)
        writer = pd.ExcelWriter('Term-Doc Matrix.xlsx', engine='xlsxwriter')
        self.matrix.to_excel(writer, 'Sheet1')
        writer.save()

    def top_cf(self, n):
        # Get the collection frequency, in order of appearance of terms in time series.
        cf_sum = self.matrix.sum(axis=1).tolist()
        col_fre = dict(zip(self.term_list, cf_sum))
        sorted_c_f = sorted(col_fre.items(), key=operator.itemgetter(1))
        i = 1
        while i <= n:
            print(sorted_c_f[-i])
            i += 1

    def top_df(self, n):
        # Get the document frequency, in order of appearance of terms in time series.
        df_sum = self.matrix.astype(bool).sum(axis=1).tolist()
        doc_fre = dict(zip(self.term_list, df_sum))
        sorted_d_f = sorted(doc_fre.items(), key=operator.itemgetter(1))
        i = 1
        while i <= n:
            print(sorted_d_f[-i])
            i += 1

    def print_result(self):
        # Print all required outputs.
        print('\n----   ----   ----   ----   ----   ----   ----   ----')
        print('***          All pages in the test data:          ***')
        for url in self.url_all:
            print(url)
        print('----   ----   ----   ----   ----   ----   ----   ----\n')

        # Print all valid pages.
        print('\n----   ----   ----   ----   ----   ----   ----   ----')
        print('***       All valid pages in the test data:       ***')
        for d in self.docs:
            print('Title:' + str(d.get_title()))
            print('Doc#' + str(d.get_id()) + ':' + str(d.get_url()))
        print('----   ----   ----   ----   ----   ----   ----   ----\n')

        # Print all visited links.
        print('\n----   ----   ----   ----   ----   ----   ----   ----')
        print('***      All visited links in the test data:       ***')
        for url in self.url_visited:
            print(url)
        print('----   ----   ----   ----   ----   ----   ----   ----\n')

        # Print all out-going links.
        print('\n----   ----   ----   ----   ----   ----   ----   ----')
        print('***       All go out links in the test data:       ***')
        for url in self.url_out:
            print(url)
        print('----   ----   ----   ----   ----   ----   ----   ----\n')

        # Print all URLs which refer to already seen contents.
        print('\n----   ----   ----   ----   ----   ----   ----   ----')
        print('***    All URLs refer to already seen contents:    ***')
        for url in self.url_already_seen:
            print(url)
        print('----   ----   ----   ----   ----   ----   ----   ----\n')

        # Print all broken links.
        print('\n----   ----   ----   ----   ----   ----   ----   ----')
        print('***       All broken links in the test data:       ***')
        for url in self.url_broken:
            print(url)
        print('----   ----   ----   ----   ----   ----   ----   ----\n')

        # Print all URLs of graphic files.
        print('\n----   ----   ----   ----   ----   ----   ----   ----')
        print('***   All URLs of graphic files in the test data:   ***')
        for url in self.url_image:
            print(url)
        print('----   ----   ----   ----   ----   ----   ----   ----\n')

        # Print all URLs of other files.
        print('\n----   ----   ----   ----   ----   ----   ----   ----')
        print('***   All URLs of other files in the test data:   ***')
        for url in self.url_other:
            print(url)
        print('----   ----   ----   ----   ----   ----   ----   ----\n')

        # Print all URLs of out of type files.
        print('\n----   ----   ----   ----   ----   ----   ----   ----')
        print('*** All URLs of out of type files in the test data:***')
        for url in self.url_outof_type:
            print(url)
        print('----   ----   ----   ----   ----   ----   ----   ----\n')

        # Print the 20th most common words with its collection frequency
        print('\n----   ----   ----   ----   ----   ----   ----   ----')
        print('***       20th most common words with its cf       ***')
        self.top_cf(20)
        print('----   ----   ----   ----   ----   ----   ----   ----\n')

        # Print the 20th most common words with its document frequency
        print('\n----   ----   ----   ----   ----   ----   ----   ----')
        print('***       20th most common words with its df       ***')
        self.top_df(20)
        print('----   ----   ----   ----   ----   ----   ----   ----\n')

    # def Euclidian_distance(self, vector1, vector2):
    #     return np.sum(np.square(np.array((vector1 - vector2))))

    def set_vectors(self):
        df = self.matrix.astype(bool).sum(axis=1).astype(float).as_matrix()
        df = np.log10(len(self.docs) / np.abs(df))
        df_vector = pd.Series(data=df, index=self.term_list)
        for d in self.docs:
            column_name = 'Doc#' + str(d.get_id())
            t_v = self.matrix[column_name]
            vector = t_v.multiply(df_vector)
            vector = vector / np.sqrt((np.sum(np.square(np.array(vector.tolist())))))
            d.set_vector(vector)

    def cos_score(self, vector1, vector2):
        dot_product = vector1.dot(vector2)
        return dot_product

    def choose_leader(self, N):
        index_set = set()
        while len(index_set) < N:
            index_set = index_set.union({random.randint(0, len(self.docs)-1)})
        for index in index_set:
            self.leader_docs.append(self.docs[index])
        self.leader_docs = sorted(self.leader_docs, key=lambda x: x.get_id())
        for d in self.leader_docs:
            self.leader_follower[d] = set()

    def classify_follower(self):
        follow_score = []
        for follower in self.docs:
            l = []
            for leader in self.leader_docs:
                l.append(self.cos_score(leader.get_vector(), follower.get_vector()))
            if max(enumerate(l), key=operator.itemgetter(1))[1] == 0:
                its_leader = (min(self.leader_follower.items(), key=lambda x: len(x[1]))[0])
            else:
                its_leader = self.leader_docs[max(enumerate(l), key=operator.itemgetter(1))[0]]
            self.leader_follower[its_leader] = self.leader_follower[its_leader].union({follower})
            follow_score.append(l)
        doc_id = []
        leader_doc_id = []
        for d in self.docs:
            doc_id.append('Doc#' + str(d.get_id()))
        for d in self.leader_docs:
            leader_doc_id.append('Doc#' + str(d.get_id()))

        # Write to a excel file.
        self.leader_matrix = pd.DataFrame(data=follow_score, index=doc_id, columns=leader_doc_id)
        writer = pd.ExcelWriter('Leader_Follower_Score Matrix.xlsx', engine='xlsxwriter')
        self.leader_matrix.to_excel(writer, 'Sheet1')
        writer.save()

        for leader in self.leader_follower:
            follower_set = self.leader_follower[leader]
            follower_set = sorted(follower_set, key=lambda x: x.get_id())
            self.leader_follower[leader] = follower_set

    def print_leader_follower(self):
        for leader in self.leader_follower.keys():
            print('\n\nLeader ID is: ', leader.get_id(), '. URL is: ', leader.get_url())
            print('It\'s followers are: ')
            for follower in self.leader_follower[leader]:
                if follower == leader:
                    continue
                print('Follower id is: ', follower.get_id(), '. URL is: ', follower.get_url(),
                      '. Score is: ', self.cos_score(leader.get_vector(), follower.get_vector()))

    def handle_documents(self):
        # Set each term vector to each documents.
        self.set_vectors()
        self.choose_leader(5)
        self.classify_follower()
        self.print_leader_follower()


if __name__ == '__main__':
    my_root_url = 'http://lyle.smu.edu/~fmoore'
    my_crawler_name = 'mycrawler'
    my_limit = 60
    my_stop_words = get_stop_words('english')
    my_scheme = ['http', 'https']
    my_netloc = ['lyle.smu.edu', 's2.smu.edu']
    myCrawler = Crawlers(my_root_url, my_crawler_name, my_limit, my_stop_words, my_scheme, my_netloc)
    myCrawler.crawl()
    myCrawler.td_matrix()
    myCrawler.print_result()
    myCrawler.handle_documents()

    myEngine = Engine(6, my_stop_words, myCrawler.term_list,
                      myCrawler.matrix.astype(bool).sum(axis=1).tolist(), myCrawler.docs)
