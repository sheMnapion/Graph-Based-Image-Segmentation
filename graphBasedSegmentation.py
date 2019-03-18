import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from unionFind import UnionSet
# This file implements 2004 Efficient Graph-Based Image Segmentation paper by
# Pedro F. Felzenszwalb and Daniel P.Huttenlocher
import time

class GraphSegmenter(object):
    """Implement the graph-based match solver"""
    
    def __init__(self,img,k=300,sigma=0.8):
        print "Initialization begin time:", time.ctime()
        self.originalImg=cv.GaussianBlur(img,(5,5),sigma)
        self.k=float(k) # tau threshold function usage
        self.loadImg(self.originalImg)
        print "Initialization end time:", time.ctime()

    def _dist(self,pixel1,pixel2):
        if len(self.originalImg.shape)==2: # gray image
            if pixel1>pixel2:
                return pixel1-pixel2
            else:
                return pixel2-pixel1
        # now RGB image
        midR=(float(pixel1[2])+float(pixel2[2]))/2.0
        deltaR=float(pixel1[2])-float(pixel2[2])
        deltaG=float(pixel1[1])-float(pixel2[1])
        deltaB=float(pixel1[0])-float(pixel2[0])
        return np.sqrt((2+midR/256.0)*deltaR**2+4*deltaG**2+(2+(255-midR)/256)*deltaB**2)

    def loadImg(self,img):
        w,h=img.shape[:2]
        self.vertices=UnionSet(w*h)
        edges=[]
        dx=[1,1,1,0]
        dy=[-1,0,1,1]
        for i in range(w):
            for j in range(h):
                for k in range(4):
                    x=i+dx[k]
                    y=j+dy[k]
                    if x<0 or x>=w or y<0 or y>=h:
                        continue
                    # print "Adding edges from (%d,%d)->(%d,%d) within [%d*%d]" % (i,j,x,y,w,h)
                    v1=i*h+j; v2=x*h+y
                    assert(v1>=0 and v1<w*h and v2>=0 and v2<w*h)
                    edges.append([i*h+j,x*h+y,self._dist(img[i][j],img[x][y])])
        self.edges=sorted(edges,key=lambda x:x[2])
        self.internalDegree=np.zeros(w*h)
        self.clusterSize=np.ones(w*h)

    def _MInt(self,xWeight,xSize,yWeight,ySize):
        return min(xWeight+float(self.k)/xSize,yWeight+float(self.k)/ySize)

    def segmentShow(self):
        """Showing segmentation results with top 90% pixels covered"""
        w,h=self.originalImg.shape[:2]
        clusterMap=np.zeros((w,h),dtype=np.uint8)
        clusterCount=np.zeros((w*h,2))
        for i in range(w*h):
            clusterCount[i][1]=i
        for i in range(w*h):
            clusterCount[self.vertices.find(i)][0]+=1
        sortedClusters=sorted(clusterCount,key=lambda x:x[0],reverse=True)
        #print sortedClusters
        validClusters=[sc for sc in sortedClusters if sc[0]>0]
        clusterProportions=[0]*len(validClusters)
        for i, vc in enumerate(validClusters):
            if i==0:
                clusterProportions[i]=float(vc[0])/(w*h)
            else:
                clusterProportions[i]=float(vc[0])/(w*h)+clusterProportions[i-1]
        ids=0
        for i, cp in enumerate(clusterProportions):
            if cp>0.90:
                print "90 percentage coverage valid ids:", i+1
                ids=i+1
                break
        validSetID=[vc[1] for vc in validClusters[:ids]]
        print "Total valid ids:", len(validSetID)
        for i in range(w):
            for j in range(h):
                if self.vertices.find(i*h+j) in validSetID:
                    clusterMap[i][j]=1+validSetID.index(self.vertices.find(i*h+j))
        plt.subplot(2,1,1); plt.imshow(clusterMap,cmap='jet')
        plt.subplot(2,1,2)
        if len(self.originalImg)==3: # RGB picture
            showImg=cv.cvtColor(self.originalImg,cv.COLOR_BGR2RGB)
            plt.imshow(showImg)
        else:
            plt.imshow(self.originalImg,cmap='gray')
        plt.savefig('Segmentation Result.jpg'); plt.show()
        return clusterMap
    
    def simpleShow(self):
        """Simply show the segmentation result without filtering"""
        w,h=self.originalImg.shape[:2]
        clusterMap=np.zeros((w,h),dtype=np.int)
        for i in range(w):
            for j in range(h):
                clusterMap[i][j]=self.vertices.find(i*h+j)
        plt.subplot(2,1,1); plt.imshow(clusterMap,cmap='jet')
        plt.subplot(2,1,2)
        if len(self.originalImg)==3: # RGB picture
            showImg=cv.cvtColor(self.originalImg,cv.COLOR_BGR2RGB)
            plt.imshow(showImg)
        else:
            plt.imshow(self.originalImg,cmap='gray')
        plt.show()

    def segment(self):
        """Perform segmentation"""
        # self.segmentShow()
        mergeCount=0
        w1,h1=self.originalImg.shape[:2]
        for (u,v,w) in self.edges:
            x1=u/h1
            y1=u%h1
            x2=u/h1
            y2=u%h1
            assert(abs(x1-x2)<=1 and abs(y1-y2)<=1)
            if self.vertices.find(u)==self.vertices.find(v):
                continue
            # print u,v,w
            uID=self.vertices.find(u)
            vID=self.vertices.find(v)
            assert(uID!=vID)
            uWeight=self.internalDegree[uID]
            uSize=self.clusterSize[uID]
            vWeight=self.internalDegree[vID]
            vSize=self.clusterSize[vID]
            if w<self._MInt(uWeight,uSize,vWeight,vSize):
                self.vertices.union(u,v)
                assert(self.vertices.find(u)==self.vertices.find(v))
                unionID=self.vertices.find(u)
                mergeCount+=1
                self.internalDegree[unionID]=max(w,uWeight,vWeight)
                # print "Partial weight:", self.internalDegree[unionID]
                self.clusterSize[unionID]=uSize+vSize
                # print "Cluster size:", self.clusterSize[unionID]
        print "Merged times %d of total pixels [%d*%d] (%.3f)" % (mergeCount,w1, \
            h1,float(mergeCount)/(w1*h1))
        return self.segmentShow()
        # self.simpleShow()

if __name__=='__main__':
    img=cv.imread('testPictures/temple_02.jpg')
    grayImg=cv.imread('testPictures/temple_02.jpg',0)
    gs=GraphSegmenter(grayImg,2500,0.8)
    cv.imwrite('output.jpg',gs.segment())