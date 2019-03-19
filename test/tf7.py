# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

x = ['c', 'a', 'd', 'b']
y = [1, 2, 3, 4]

plt.bar(x, y, color="yellow" )

# plt.show()

word_index = {"zhangbo": 24, 
                        "zhangwei":35,
                        "zhangguang":23}

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

print(reverse_word_index)