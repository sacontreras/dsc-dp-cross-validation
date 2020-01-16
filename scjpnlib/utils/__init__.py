from functools import reduce
import operator as op
from itertools import combinations
import pandas as pd
import numpy as np
from IPython.core.display import HTML

# credit to Bunny Rabbit for code: Rabbit, B. (2018). Revisions to Is there a math nCr function in python? [duplicate]. Retrieved rom https://stackoverflow.com/posts/4941932/revisions
def nCr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return int(numer / denom)

# this function allows displaying dataframes using print() with pandas "pretty" HTML formatting
#   so that multiple "pretty" displays of dataframes can be rendered "inline"
#   default behavior (without specifying a range) is identical to that of df.head()
def print_df(df, n = None, tail=False):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', -1)

    if n is None:
        n = len(df)
    display(HTML(df.head(n).to_html() if not tail else df.tail(n).to_html()))
    display(HTML("<br>{} rows x {} columns<br><br>".format(n, len(df.columns))))

    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_colwidth')