# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 00:19:50 2022

@author: Balasubramaniam
"""
'''
python -m spacy download en_core_web_sm
'''
import spacy
nlp = spacy.load("en_core_web_sm")
text = "It’s official: Apple is the first U.S. public company to reach a $1 trillion market value"
 # Process the text
doc = nlp(text)
for token in doc:
     # Get the token text, part-of-speech tag and dependency label
     token_text = token.text
     token_pos = token.pos_
     token_dep = token.dep_
     # This is for formatting only
     print(f"{token_text:<12}{token_pos:<10}{token_dep:<10}") 
     
     
