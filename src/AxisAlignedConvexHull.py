import numpy as np

######## main implementation ########
        
class AxisAlignedConvexHull:
    ## a constructor with parameter
    def __init__(self, iterN):
        coeff = np.eye(iterN * 2, 2)
        
        for index in range(1, iterN):
            coeff[2 * index, 0] = (iterN - index)
            coeff[2 * index, 1] = index
            
            coeff[(2 * index) + 1, 0] = iterN - index
            coeff[(2 * index) + 1, 1] = -index

        self.coeff = np.array(coeff)
        self.boundary = np.zeros(self.coeff.shape)
        self.center = np.zeros((2,))

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

    ## true if pos is inside of this
    def isIn(self, pos):
        for index in range(0, self.coeff.shape[0]):
            value = np.inner(pos - self.center, self.coeff[index, :])
            if (value < self.boundary[index, 0]):
                return False
            elif (value > self.boundary[index, 1]):
                return False

        return True

######## functions for the unit test ########
        
## vertices for visualization
def getVertices(boundaryBox):
    previousLineIndex = 0
    vertices = []

    angleList = np.linspace(0.1, 361.1, 600)
        
    for angleDegree in angleList:
        minLineIndex = 0
        minLineDistance = 1.0e100

        angle = angleDegree/180.0*(np.pi)
            
        unitVector = np.array([np.cos(angle), np.sin(angle)])
        for coeffIndex in range(0, boundaryBox.coeff.shape[0]):
            distance = 0.0
                
            if np.inner(unitVector, boundaryBox.coeff[coeffIndex, :]) > 0.0:
                distance = distanceAlongDirection(
                    boundaryBox.coeff[coeffIndex, 0], boundaryBox.coeff[coeffIndex, 1],
                    -boundaryBox.boundary[coeffIndex, 1],
                    unitVector
                )
            else: 
                distance = distanceAlongDirection(
                    -boundaryBox.coeff[coeffIndex, 0], -boundaryBox.coeff[coeffIndex, 1],
                    boundaryBox.boundary[coeffIndex, 0],
                    unitVector
                )

            if distance < minLineDistance:
                minLineDistance = distance
                minLineIndex = coeffIndex

        if (minLineIndex != previousLineIndex):
            vertex = minLineDistance * unitVector
            vertices = vertices + [vertex]
            previousLineIndex = minLineIndex

    vertices = vertices + [vertices[0]]

    return np.array(vertices) + boundaryBox.center
  
## a distance metric for calculating vertices
def distanceAlongDirection(A, B, c, direction):
    if abs((A * direction[0]) + (B * direction[1])) < 1e-8:
        return abs(c) / np.sqrt((A * A) + (B * B))
    else:
        return abs(c) / ((A * direction[0]) + (B * direction[1]))

######## unit test ########
        
if __name__ == '__main__':
    resol = 5
    
    gridSize = 100
    
    pointNum = 12

    # random input generation
    testPoints = np.random.random((pointNum, 2)) * np.random.random() + np.random.random((1, 2))

    # box generation
    testBox = AxisAlignedConvexHull(resol)
    testBox.generate(testPoints)

    # vertices for visualization
    vertrices = getVertices(testBox)

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

    plt.plot(inPoint[:, 0], inPoint[:, 1], ' g.')
    plt.plot(outPoint[:, 0], outPoint[:, 1], ' k.')
    plt.plot(vertrices[:, 0], vertrices[:, 1], 'b-')
    plt.plot(testPoints[:, 0], testPoints[:, 1], 'ro')

    plt.title(
f'''Axis-Aligned Convex-Hull
# of boundary sets: {testBox.boundary.shape[0]//2}
# of params: {testBox.boundary.shape[0] * 2}'''
              )
    
    plt.xlim(xRange)
    plt.ylim(yRange)

    plt.legend(
        ['Inside', 'Outside', 'AACH boundary', 'Input points'],
        loc='upper right'
        )
