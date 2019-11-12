import os
import re
import numpy as np
from mrjob.job import MRJob
from mrjob.step import MRStep
from operator import itemgetter

word_search_re = re.compile(r"[\w']+")


class NaiveBayesTrainer(MRJob):
    
    def steps(self):
        return [
            MRStep(mapper=self.extract_words_mapping,
                   reducer=self.reducer_count_words),
            MRStep(reducer=self.compare_words_reducer),
            ]

    def extract_words_mapping(self, key, value):
        tokens = value.split()
        gender = eval(tokens[0])
        blog_post = eval(" ".join(tokens[1:]))
        all_words = word_search_re.findall(blog_post)
        all_words = [word.lower() for word in all_words]
        for word in all_words:
            yield (gender, word), 1. / len(all_words)

    def reducer_count_words(self, key, counts):
        s = sum(counts)
        gender, word = key
        yield word, (gender, s)

    def compare_words_reducer(self, word, values):
        per_gender = {}
        for value in values:
            gender, s = value
            per_gender[gender] = s
        yield word, per_gender

if __name__ == '__main__':
    NaiveBayesTrainer.run()
