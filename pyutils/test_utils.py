# ====================================================================================== #
# Forest test utils.py
# Author : Eddie Lee, edlee@santafe.edu
# 
#
# MIT License
# 
# Copyright (c) 2021 Edward D. Lee
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE.
# ====================================================================================== #
from .utils import *
from itertools import combinations
from scipy.integrate import quad



def test_row_ix_from_utri():
    for n in range(2, 6):
        pairs = list(combinations(range(n), 2))
        for i in range(n):
            ix = row_ix_from_utri(i, n)
        for ix_ in ix:
            assert i in pairs[ix_], (i, pairs[ix_])

