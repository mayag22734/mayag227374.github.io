## Data Generation with Neural Network

**Project description:** Using PyQt5 and PyQtGraph, the entire program can be handled via GUI. Once the data (points with X,Y coordinates and an assigned class) is generated and the network configured, training can be done utilizing variable learning rate in 400-epoch chunks. The neural network is a very simple one consisting of one hidden layer with a variable number of neurons inside the hidden layer implemented using linear and activation layers and a sigmoid activation function.

### Implementation
```Python
import sys
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QFont
import numpy as np
import pyqtgraph as pg
import math

pg.setConfigOption('background', '#37383b')
pg.setConfigOption('foreground', 'w')

class Window(qtw.QDialog):
	def __init__(self, parent=None):
		super(Window, self).__init__(parent)
		self.initLabels()
		self.initInputs()
		self.initOtherWidgets()
		self.initLayout()
		self.networkMade = False

	def initLabels(self):
		self.labelModes = qtw.QLabel('Number of Modes (up to 10)')
		self.labelSamples = qtw.QLabel('Number of Samples (up to 100)')
		self.labelNeurons = qtw.QLabel('Number of neurons in hidden layer (up to 100)')
		self.labelBatch = qtw.QLabel('Batch size (up to 100)')
		for label in (self.labelModes, self.labelSamples, self.labelNeurons, self.labelBatch):
			label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
			label.setFont(QFont('Times',10))

	def initInputs(self):
		self.modeInput = qtw.QSpinBox()
		self.sampleInput = qtw.QSpinBox()
		self.neuronInput = qtw.QSpinBox()
		self.batchInput = qtw.QSpinBox()
		inputs = [self.modeInput,self.sampleInput,self.neuronInput,self.batchInput]
		for input in inputs:
			input.setMinimum(1)
			input.setMaximum(100)


	def initOtherWidgets(self):
		self.fig = pg.PlotWidget()
		self.plotButton = qtw.QPushButton('Plot')
		self.plotButton.clicked.connect(lambda:self.makeData())
		self.neuronButton = qtw.QPushButton('Make New Network')
		self.neuronButton.clicked.connect(lambda:self.makeNetwork())
		self.trainButton = qtw.QPushButton('Train Network')
		self.trainButton.clicked.connect(lambda:self.trainNetwork())

	def initLayout(self):
		self.layout = qtw.QVBoxLayout()
		self.setLayout(self.layout)
		self.layout.addWidget(self.fig)
		self.layout.addWidget(self.labelModes)
		self.layout.addWidget(self.modeInput)
		self.layout.addWidget(self.labelSamples)
		self.layout.addWidget(self.sampleInput)
		self.layout.addWidget(self.labelNeurons)
		self.layout.addWidget(self.neuronInput)
		self.layout.addWidget(self.labelBatch)
		self.layout.addWidget(self.batchInput)
		self.buttonsLayout = qtw.QHBoxLayout()
		self.buttonsLayout.addWidget(self.plotButton)
		self.buttonsLayout.addWidget(self.neuronButton)
		self.buttonsLayout.addWidget(self.trainButton)
		self.layout.addLayout(self.buttonsLayout)

	def makeNetwork(self):
		self.network = Network(3, self.samples, self.neuronInput.value(), self.batchInput.value())
		self.networkMade = True

	def makeData(self):
		self.modeNo = self.modeInput.value()
		self.sampleNo = self.sampleInput.value()
		self.setA = DataSet(0,self.modeNo,self.sampleNo)
		self.setB = DataSet(1,self.modeNo,self.sampleNo)
		self.samples = np.reshape(np.append(self.setA.modeList,self.setB.modeList),(self.modeNo*self.sampleNo*2,3))
		self.fig.enableAutoRange()
		self.fig.clear()
		self.drawPoints()
		self.newPlot = True

	def drawPoints(self):
		self.fig.clear()
		self.fig.plot(self.setA.modeList[:,0],self.setA.modeList[:,1],pen=None,symbol='o',symbolPen=pg.mkPen('r'),symbolSize=5, symbolBrush=pg.mkBrush('r'))
		self.fig.plot(self.setB.modeList[:,0],self.setB.modeList[:,1],pen=None,symbol='o',symbolPen=pg.mkPen('b'),symbolSize=5, symbolBrush=pg.mkBrush('b'))

	def drawBoundaries(self):
		boundPoints = 100
		samplesX = []
		samplesY = []
		view = self.fig.viewRange()
		viewRect = self.fig.viewRect()
		self.fig.clear()
		self.drawPoints()
		self.fig.disableAutoRange()
		x,y = np.meshgrid(np.linspace(view[0][0],view[0][1],boundPoints),np.linspace(view[1][0],view[1][1],boundPoints))
		boundary = np.empty((boundPoints,boundPoints,2))
		for i in range(0,boundPoints):
			boundary[i] = self.network.calculatePoint(np.c_[x[i],y[i]])
		self.boundaryImg = [pg.ImageItem(),pg.ImageItem()]
		pos = [np.array([0.,1.]),np.array([1.,0.])]
		for i in range(0,2):
			self.boundaryImg[i].setImage(boundary[:,:,i])
			self.boundaryImg[i].setRect(viewRect)
			color = np.array([[[240, 170, 170,64],[175, 173, 240, 64]],[[175, 173, 240, 64],[240, 170, 170,64]]])
			self.fig.addItem(self.boundaryImg[i])
			cmap = pg.ColorMap(pos[i],color[i])
			lut = cmap.getLookupTable()
			self.boundaryImg[i].setLookupTable(lut)
		

	def trainNetwork(self):
		if self.networkMade:
			self.network.trainNetwork()
			self.drawBoundaries()

class DataSet():
	def __init__(self, label = 0, modes = 1, sampleNo = 1, parent=None):
		self.label = label
		self.modes = modes
		self.sampleNo = sampleNo
		self.modeList = []
		for i in range(0,self.modes):
			self.modeList.append(self.generateMode())
		self.modeList = np.array(np.reshape(self.modeList,(self.sampleNo*self.modes,2)))
		self.modeList = np.c_[self.modeList,self.label*np.ones((sampleNo*modes))]

	def generateMode(self):
		rng = np.random.default_rng()
		mean = rng.uniform(-1,1)
		dev = rng.uniform(0,0.25)
		dataX = rng.normal(mean,dev,self.sampleNo)
		dataY = rng.normal(mean,dev,self.sampleNo)
		return np.c_[np.reshape(dataX,(self.sampleNo,1)),np.reshape(dataY,(self.sampleNo,1))]

class Network():
	def __init__(self, layers, points, hiddenNeuronNo, batchSize):
		self.batchSize = batchSize
		self.layers = [Linear(2,2),Activation(batchSize),Linear(hiddenNeuronNo,2),Activation(batchSize),Linear(2,hiddenNeuronNo),Activation(batchSize)]
		self.points = points

	def calculatePoint(self, point):
		for layer in self.layers:
			point = layer.forward(point)
		return point

	def trainNetwork(self):
		rateMax = 0.1
		rateMin = 0.0001
		epochs = 400
		e = 0
		ePerCycle = 50
		while True:
			if e >= epochs:
				break
			else:
				np.random.shuffle(self.points)
				labels = self.points[:,2]
				cosInner = (math.pi*(e % ePerCycle))/(ePerCycle)
				rate = rateMax/2 * (math.cos(cosInner)+1)
				for i in range(0,len(self.points),self.batchSize):
					output = self.points[i:i+self.batchSize,:2]
					for layer in self.layers:
						output = layer.forward(output)
					grad = 2*(np.expand_dims(labels[i:i+self.batchSize],-1)-output)
					for layer in self.layers[::-1]:
						grad = layer.backward(grad)
					for layer in self.layers:
						layer.adjust(rate)
				e += 1
		print("Training finished")

class Linear():
	def __init__(self, neuronNo, inputNo):
		self.points = []
		self.weights = []
		self.inputNo = inputNo
		self.neuronNo = neuronNo
		for neuron in range(0,neuronNo):
			self.weights.append(np.random.randn(inputNo+1))
		self.weights = np.array(self.weights)
	
	def forward(self, points):
		self.points = np.c_[np.ones(len(points))*-1,points]
		return np.matmul(self.points,np.transpose(self.weights))

	def backward(self, grad):
		self.grad = grad
		gradW = np.matmul(grad,self.weights[:,:len(self.weights[0])-1])
		return gradW

	def adjust(self, rate):
		delta = rate * np.matmul(np.transpose(self.grad),self.points)
		self.weights += delta

class Activation():
	def __init__(self, batchSize):
		self.states = np.empty(batchSize)

	def forward(self, stateList):
		self.states = stateList
		out = self.sigmoid(stateList)
		return out

	def backward(self, grad):
		return self.sigmoidDv(self.states) * grad

	def adjust(self, rate):
		pass

	def sigmoidDv(self, states):
		return self.sigmoid(states)*(np.ones((len(states),len(states[0])))-self.sigmoid(states))

	def sigmoid(self, states):
		return 1/(1+np.exp(-states))

if __name__ == "__main__":
	app = qtw.QApplication(sys.argv)
	main = Window()
	main.show()
	sys.exit(app.exec_())
```
