import sys
import os
sys.path.append("T:\\megadata")

from helper.regrassion import *
from helper.util import *
from helper.plot import *
from helper.analysis import *
import joblib

fit = joblib.load(r"T:\\megadata\\F.추론통계(머신러닝)\04.지도학습-회귀\\test\\mymodel.pkl")
y = fit.predict([[1, 160, 1, 1]])
print(y)