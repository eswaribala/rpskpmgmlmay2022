# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 11:02:13 2022

@author: Balasubramaniam
"""
from abc import ABC, abstractmethod

class KYC(ABC):
    @abstractmethod
    def getDocuments(self,documents):pass
    @abstractmethod  
    def validateDocuments(self,documents):pass
    
    
class HSBCKYC(KYC):
    
    def __init__(self,bankName):
        print("Constructor ready...")
        self.__bankName=bankName
        
    
    def getDocuments(self, documents):
        self.__documents=documents
        print("Docs received")
    
    #setters and getters
        
    def setBankName(self,bankName):
        self.__bankName=bankName

    def getBankName(self):
        return self.__bankName       
    def validateDocuments(self,documents):
        print("Validation Starts....")
        

hsbcKYC=HSBCKYC("HSBC")
hsbcKYC.validateDocuments("document")
print(hsbcKYC.getBankName())
        
        
