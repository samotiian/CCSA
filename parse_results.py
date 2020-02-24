import re
import numpy as np

def parse_results():

    with open('results.txt', 'r') as f:
        txt = f.readlines()

    search_string = r"Best accuracy for ([\d]) target sample per class and repetition ([\d]) is ([\d.]+)."
    
    results = np.array(
        list(
            map(
                lambda x: float(x.group(3)),
                filter(
                    lambda x: bool(x),
                    map(
                        lambda x: re.search(search_string, x),
                        txt
                    )
                )
            )
        )
    )


    for i, samples in enumerate([1,3,5,7]):
        res = results[10*i:10*(i+1)]
        # print(res)
        print(f"{samples} samples: mean = {res.mean()}, std = {res.std()}")

if __name__ == '__main__':
    parse_results()
