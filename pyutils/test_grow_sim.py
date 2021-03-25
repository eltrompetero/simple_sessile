# ====================================================================================== #
# Automata compartment model for forest growth.
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
from .grow_sim import *



def test_overlap_area():
    assert overlap_area(2, 1, 1)==0
    assert overlap_area(2, 1, 1)==0
    assert overlap_area(0, 1, 1)==np.pi

def test_delete_flat_dist_rowcol():
    np.random.seed(0)
    n = 10

    d = np.random.rand(n * (n-1) // 2)
    dsquare = squareform(d)
    
    for i in range(n):
        newd = delete_flat_dist_rowcol(d, i, n)
        newdsquare = np.delete(np.delete(dsquare, i, axis=0), i, axis=1)
        assert np.array_equal(squareform(newd), newdsquare)

def test_append_flat_dist_rowcol(n=10):
    np.random.seed(0)

    d = np.random.rand(n * (n-1) // 2)
    print(d)
    newd = delete_flat_dist_rowcol(append_flat_dist_rowcol(d, -1, n), n, n+1)
    print(newd)
    assert np.array_equal(d, newd)
