# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 12:48:59 2020

@author: gkhnmlym
"""


#.ui uzantılı dosyayı aşağıdaki kodla .py uzantısına dönüştürürüz.
from PyQt5 import uic

with open('arayuzui.py', 'w', encoding="utf-8") as fout:
   uic.compileUi('arayuz.ui', fout)
