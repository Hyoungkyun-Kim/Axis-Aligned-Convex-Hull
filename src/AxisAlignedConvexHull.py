import sys
sys.path.insert(1, './action/')
sys.path.insert(2, './character/')
sys.path.insert(3, './collision/')
sys.path.insert(4, './physics/')
sys.path.insert(5, './platform/')

import numpy as np
import queue
from threading import Thread

######## main implementation ########
        
class AxisAlignedConvexHull:
    ## a constructor with parameter
    def __init__(self, iterN, precision):
        self.precision = precision
        coeff = np.eye(iterN * 2, 2)
        
        for index in range(1, iterN):
            coeff[2 * index, 0] = iterN - index
            coeff[2 * index, 1] = index
            
            coeff[(2 * index) + 1, 0] = index
            coeff[(2 * index) + 1, 1] = index - iterN

        self.coeff = np.array(coeff)
        self.boundary = np.zeros(self.coeff.shape)
        self.center = np.zeros((2,))

        self.vertices = self.getVertices()
        
    ## generating this from points
    def generate(self, points):
        self.center = 0.5 * (points.max(axis=0) + points.min(axis=0))

        error = points - self.center
        
        boundary = np.ones((self.coeff.shape[0], 2))

        for index in range(0, self.coeff.shape[0]):
            projection = np.inner(error, self.coeff[index, :])
            boundary[index, 0] = projection.min()
            boundary[index, 1] = projection.max()
        
        self.boundary = boundary
        self.vertices = self.getVertices()

    ## true if pos is inside of this
    def isIn(self, pos, margin = 0.0):
        for index in range(0, self.coeff.shape[0]):
            value = np.inner(pos - self.center, self.coeff[index, :])
            if (value < (self.boundary[index, 0] - self.precision - margin)):
                return False
            elif (value > (self.boundary[index, 1] + self.precision + margin)):
                return False

        return True

    ## get boundary vertices
    def getVertices(self):
        verticesQueue = queue.Queue()

        vertexThread = []
        for index0 in range(0, self.coeff.shape[0]):
            for index1  in range(index0 + 1, self.coeff.shape[0]):
                vertexThread = vertexThread + [Thread(target=self.getVertexThread, args=(verticesQueue, index0, index1))]
                vertexThread[-1].start()
                
        for index in range(0, len(vertexThread)):
            vertexThread[index].join()

        vertices = np.array(verticesQueue.queue)
        
        vertices = ((vertices / (self.precision * self.precision)) // (1 / self.precision)) * self.precision
        vertices = np.unique(vertices, axis=0)
        
        sortedIndices = np.argsort(np.arctan2(vertices[:, 0], vertices[:, 1]))
        sortedIndices = np.concatenate((sortedIndices, [sortedIndices[0]]))

        return vertices[sortedIndices] + self.center

    def getVertexThread(self, verticesQueue, index0, index1):
        A = np.vstack((self.coeff[index0, :], self.coeff[index1, :]))
        
        B = np.array([self.boundary[index0, 0], self.boundary[index1, 0]]).transpose()
        aThread = [Thread(target=self.getSigleVertexThread, args=(verticesQueue, A, B))]
        aThread[-1].start()
        
        B = np.array([self.boundary[index0, 0], self.boundary[index1, 1]]).transpose()
        aThread = aThread + [Thread(target=self.getSigleVertexThread, args=(verticesQueue, A, B))]
        aThread[-1].start()

        B = np.array([self.boundary[index0, 1], self.boundary[index1, 0]]).transpose()
        aThread = aThread + [Thread(target=self.getSigleVertexThread, args=(verticesQueue, A, B))]
        aThread[-1].start()
        
        B = np.array([self.boundary[index0, 1], self.boundary[index1, 1]]).transpose()
        aThread = aThread + [Thread(target=self.getSigleVertexThread, args=(verticesQueue, A, B))]
        aThread[-1].start()

        for index in range(0, len(aThread)):
            aThread[index].join()
        
        
    def getSigleVertexThread(self, verticesQueue, A, B):
        intersections = np.inner(np.linalg.pinv(A), B)
        if self.isIn(intersections + self.center):
            verticesQueue.put(intersections)
        
######## unit test ########

if __name__ == '__main__':
    iterN = 5
    precision = 1e-6
    
    gridSize = 100
    
    pointNum = 12

    # random input generation
    testPoints = np.random.random((pointNum, 2)) * np.random.random() + np.random.random((1, 2))

    # box generation
    testBox = AxisAlignedConvexHull(iterN, precision)
    testBox.generate(testPoints)
    
    # vertices for visualization
    vertices = testBox.getVertices()

    # grid for in-out check test
    margin = 0.75 * (testPoints.max(axis=0)[0] - testPoints.min(axis=0)[0])
    
    xRange = [testPoints.min(axis=0)[0] - margin, testPoints.max(axis=0)[0] + margin]
    yRange = [testPoints.min(axis=0)[1] - margin, testPoints.max(axis=0)[1] + margin]
    
    X = np.linspace(xRange[0], xRange[1], gridSize)
    Y = np.linspace(yRange[0], yRange[1], gridSize)

    # in-out check test
    inPoint = []
    outPoint = []

    for indexX in range(0, X.shape[0]):
        for indexY in range(0, Y.shape[0]):
            if testBox.isIn(np.array([X[indexX], Y[indexY]])):
                inPoint = inPoint + [[X[indexX], Y[indexY]]]
            else:
                outPoint = outPoint + [[X[indexX], Y[indexY]]]

    inPoint = np.array(inPoint)
    outPoint = np.array(outPoint)

    # visualize results
    import matplotlib.pyplot as plt
    plt.ion()

    plt.figure()

    plt.plot(inPoint[:, 0], inPoint[:, 1], ' w.')
    plt.plot(outPoint[:, 0], outPoint[:, 1], ' k.')
    plt.plot(vertices[:, 0], vertices[:, 1], '-', color=(1, 1, 0))
    plt.plot(testPoints[:, 0], testPoints[:, 1], 'ro')

    plt.title(
f'''Axis-Aligned Convex-Hull
# of boundary sets: {testBox.boundary.shape[0]//2}
# of params: {testBox.boundary.shape[0] * 2}
# of vertices: {vertices.shape[0]-1}'''
              )
    
    plt.xlim(xRange)
    plt.ylim(yRange)

    plt.legend(
        ['Inside', 'Outside', 'AACH boundary', 'Input points'],
        loc='upper right'
        )
    
