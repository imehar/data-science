import os
import re
from mrjob.job import MRJob
from mrjob.step import MRStep

word_search_re = re.compile(r"[\w]+")


class ExtractPosts(MRJob):

    post_start = False
    post = []

    def mapper(self, key, line):
        filename = os.environ["map_input_file"]
        gender = filename.split(".")[1]
        line = line.strip()
        if line == "<post>":
            self.post_start = True
        elif line == "</post>":
            self.post_start = False
            yield gender, str("\n".join(self.post))
            self.post = []
        elif self.post_start:
            self.post.append(line)



if __name__ == '__main__':
    ExtractPosts.run()
