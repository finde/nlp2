from __future__ import division, print_function
from pprint import pprint

def mkcorpus(sentences):
    """
        create a list of sentences pair (tuple)
    """
    return [(es.split(), fs.split()) for (es, fs) in sentences]


def matrix(
        m, n, lst,
        m_text=None,
        n_text=None):
    """
    m: row
    n: column
    lst: items

    Matrix visualisation
    """

    fmt = ""
    if n_text:
        fmt += "     {}\n".format(" ".join(n_text))
    for i in range(1, m + 1):
        if m_text:
            fmt += "{:<4.4} ".format(m_text[i - 1])
        fmt += "|"
        for j in range(1, n + 1):
            if (i, j) in lst:
                fmt += "x|"
            else:
                fmt += " |"
        fmt += "\n"
    return fmt


if __name__ == "__main__":
    sent_pairs = [("Saya adalah seorang pria", "I am a man"),
                  ("Dia adalah seorang wanita", "I am a girl"),
                  ("Saya adalah seorang guru", "I am a teacher"),
                  ("Dia adalah seorang guru", "She is a teacher"),
                  ("Dia adalah seorang guru", "He is a teacher"),
    ]

    pprint(mkcorpus(sent_pairs))
    print(matrix(5, 30, [(1, 1), (4, 15)]))