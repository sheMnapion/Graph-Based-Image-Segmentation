import numpy as np
#This file implements a union set

class UnionSet(object):
    """Union set, for disjoint set operations"""

    def __init__(self,setSize):
        """Init a union set with size [setSize]"""
        self.id=np.arange(setSize,dtype=np.int)
        self.rank=np.zeros(setSize,dtype=np.int)

    def find(self,x):
        """Find the set element [x] is in"""
        if self.id[x]!=x:
            self.id[x]=self.find(self.id[x])
        return self.id[x]

    def union(self,x,y):
        """Union two sets [x] and [y]"""
        xID=self.find(x)
        yID=self.find(y)
        if self.rank[xID]>self.rank[yID]:
            self.id[yID]=xID
        else:
            self.id[xID]=yID
            if self.rank[xID]==self.rank[yID]:
                self.rank[yID]+=1
