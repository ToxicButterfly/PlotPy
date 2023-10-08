import matplotlib.pyplot as plt
import urllib.request

import pandas as pd

from Plots import Plots

plot = Plots()
plot.draw_plots("https://ai-process-sandy.s3.eu-west-1.amazonaws.com/purge/deviation.json ")
