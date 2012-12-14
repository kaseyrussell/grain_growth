#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

# run this at a bash terminal with:
# python profile_voronoi.py

import pstats, cProfile
import numpy as np
#import voronoi
#reload(voronoi)

import test_graph_initialize

cProfile.runctx("test_graph_initialize.runtest()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()


