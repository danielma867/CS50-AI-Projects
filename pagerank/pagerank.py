import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

def crawl(directory):
    pages = dict()
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?\s+)?href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}
    return pages

def transition_model(corpus, page, damping_factor):
    model = {}
    links = corpus[page]
    n = len(corpus)
    if links:
        for p in corpus:
            model[p] = (1 - damping_factor) / n
        for link in links:
            model[link] += damping_factor / len(links)
    else:
        for p in corpus:
            model[p] = 1 / n
    return model

def sample_pagerank(corpus, damping_factor, n):
    samples = {page: 0 for page in corpus}
    page = random.choice(list(corpus.keys()))
    for _ in range(n):
        samples[page] += 1
        model = transition_model(corpus, page, damping_factor)
        page = random.choices(list(model.keys()), weights=model.values(), k=1)[0]
    return {page: count / n for page, count in samples.items()}

def iterate_pagerank(corpus, damping_factor):
    n = len(corpus)
    ranks = {page: 1 / n for page in corpus}
    while True:
        new_ranks = {}
        for page in corpus:
            new_rank = (1 - damping_factor) / n
            sum_val = 0
            for p, links in corpus.items():
                if page in links:
                    sum_val += ranks[p] / len(links)
                elif not links:
                    sum_val += ranks[p] / n
            new_rank += damping_factor * sum_val
            new_ranks[page] = new_rank
        if all(abs(new_ranks[p] - ranks[p]) < 0.001 for p in corpus):
            break
        ranks = new_ranks
    return ranks

if __name__ == "__main__":
    main()
