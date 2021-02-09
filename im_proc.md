## Command-line Image Processing Application

**Project description:** This student project involved implementation of basic image processing concepts such as morphological operations, filtering, and fourier transform and frequency-domain operations. The entire project is coded in basic Python from scratch, with all operations implemented manually using only the Pillow library for loading and saving images and Numpy for array operations and optimization. Each operation is functional for both grayscale and RGB images where appropriate and feasible within the scope of a student project. Be aware that the order of arguments determines the order of operations.

### Implementation
```python
import sys, getopt
from PIL import Image
import numpy as np
import math
import cmath
from time import perf_counter


def main(argv):
	imagePath = ''
	outPath = None

	try:
		opts, args = getopt.getopt(argv,"hi:o:b:c:n",["help","input=","output=","brightness=","contrast=","negative","median=","mse=","pmse=","maxdiff=","hflip","vflip","dflip","shrink=", "enlarge=","resize=","hmean=","snr=","psnr=","histogram","huniform", "cstdev", "cvarcoi", "slaplace=","wallis","regiongrow=", "dilate", "erode", "open", "close", 'hmt', 'm6', 'dft', 'fft', 'lowpass=', 'ifft', 'idft', 'highpass=', 'bandpass=', 'bandcut=', 'edgedetection=','addorig', 'phasemodify='])
	except getopt.GetoptError:
		print('ImProc1 --input <file> --command <argument>')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ('-h','--help'):
			print('ImProc1 --input <file> --command <argument>')
			sys.exit()
		elif opt in ('-i','--input'):
			imagePath = arg
			image = np.array(Image.open(imagePath))
			image = np.float64(image)
			#print(len(image[0][0]))
		elif opt in ('-o','--output'):
			outPath = arg
		elif opt in ('-b','--brightness'):
			image = changeBright(image, float(arg))
		elif opt in ('-c','--contrast'):
			image = changeContrast(image, float(arg))
		elif opt in ('-n','--negative'):
			image = makeNegative(image)
		elif opt in ('--hflip'):
			image = horizontalFlip(image)
		elif opt in ('--vflip'):
			image = verticalFlip(image)
		elif opt in ('--dflip'):
			image = diagonalFlip(image)
		elif opt in ('--shrink') or opt in ('--enlarge') or opt in ('--resize'):
			dim = arg.split('x')
			image = resize(image,int(dim[0]),int(dim[1]))
		elif opt in ('--median'):
			image = filterMedian(image, int(arg))
		elif opt in ('--hmean'):
			image = filterHarmonic(image,int(arg))
		elif opt in ('--mse'):
			imagePath = arg
			original = np.array(Image.open(imagePath))
			original = np.float64(original)
			print('MSE is: ' + str(getMSE(original,image)))
		elif opt in ('--pmse'):
			imagePath = arg
			original = np.array(Image.open(imagePath))
			original = np.float64(original)
			print('Peak MSE is: ' + str(getPeakMSE(original,image)))
		elif opt in ('--maxdiff'):
			imagePath = arg
			original = np.array(Image.open(imagePath))
			original = np.float64(original)
			print('Maximum Difference is: ' + str(getMD(original,image)))
		elif opt in ('--snr'):
			imagePath = arg
			original = np.array(Image.open(imagePath))
			original = np.float64(original)
			print('SNR is: ' + str(getSNR(original,image)))
		elif opt in ('--psnr'):
			imagePath = arg
			original = np.array(Image.open(imagePath))
			original = np.float64(original)
			print('Peak SNR is: ' + str(getPeakSNR(original,image)))
		elif opt in ('--histogram'):
			try:
				if (len(image[0][0] > 1)):
					histImg,hist = makeHistogram(image,True)
					histImgR = Image.fromarray(np.uint8(histImg[0]))
					histImgR.save("histogramRed.bmp")
					histImgG = Image.fromarray(np.uint8(histImg[1]))
					histImgG.save("histogramGreen.bmp")
					histImgB = Image.fromarray(np.uint8(histImg[2]))
					histImgB.save("histogramBlue.bmp")
			except TypeError:
				histImg,hist = makeHistogram(image,False)
				histImg = Image.fromarray(np.uint8(histImg))
				histImg.save("histogram.bmp")
		elif opt in ('--huniform'):
			image = hUniform(image,hist)
		elif opt in ('--slaplace'):
			if arg == 'opt':
				image = optLaplace(image)
			else:
				image = laplace(image,int(arg))
		elif opt in ('--wallis'):
			image = wallis(image)
		elif opt in ('--cstdev'):
			print('Standard Deviation is: ' + str(getStdDev(image,hist)))
		elif opt in ('--cvarcoi'):
			print('Variation Coefficient 1 is: ' + str(getStdDev(image,hist)/getMean(image,hist)))
		elif opt in ('--regiongrow'):
			#image = growRegion(image, int(arg))
			image = growRegion(image, arg.split(','))
		elif opt in ('--dilate'):
			image = dilate(image)
		elif opt in ('--erode'):
			image = erode(image)
		elif opt in ('--open'):
			image = dilate(erode(image))
		elif opt in ('--close'):
			image = erode(dilate(image))
		elif opt in ('--hmt'):
			# + = fg, - = bg
			#plus with bg center, fg neighbors
			#foremask = [(-1,0),(1,0),(0,-1),(0,1)]
			#backmask = [(0,0)]

			#fg L with bg in center
			# +++
			#foremask = [(-1,-1),(-1,0),(-1,1),(0,1)]
			#backmask = [(0,0)]

			# ++-
			# -+-
			# -+-
			foremask = [(-1,-1),(-1,0),(0,0),(1,0)]
			backmask = [(0,-1),(1,-1),(-1,1),(0,1),(1,1)]
			image = hitmiss(image, foremask, backmask)
		elif opt in ('--m6'):
			image = m6(image)
		elif opt in ('--dft'):
			ftar = dft(image)
		elif opt in ('--idft'):
			image = ift(ftar, outPath)
		elif opt in ('--fft'):
			ftar = performFft(image)
		elif opt in ('--ifft'):
			image = performIfft(ftar, outPath)
		elif opt in ('--lowpass'):
			ftar = lowPass(ftar, int(arg))
		elif opt in ('--highpass'):
			ftar = highPass(ftar, int(arg))
		elif opt in ('--bandpass'):
			params = arg.split(',')
			ftar = bandPass(ftar, int(params[0]), int(params[1]))
		elif opt in ('--bandcut'):
			params = arg.split(',')
			ftar = bandCut(ftar, int(params[0]), int(params[1]))
		elif opt in ('--edgedetection'):
			params = arg.split(',')
			ftar = detectEdge(ftar, int(params[0]), int(params[1]), int(params[2]))
		elif opt in ('--addorig'):
			orig = np.array(Image.open(imagePath))
			orig = np.float64(orig)
			if orig.shape != image.shape:
				orig = resize(orig,image.shape[1],image.shape[0])
			orig += image
			np.putmask(orig, orig>255, 255)
			np.putmask(orig, orig<0, 0)
			orig = Image.fromarray(np.uint8(orig))
			origPath = outPath.split('.')
			orig.save(origPath[0]+'Added'+'.'+origPath[1])
		elif opt in ('--phasemodify'):
			params = arg.split(',')
			ftar = phaseModify(ftar, int(params[0]), int(params[1]))

	np.putmask(image,image>255,255)
	np.putmask(image,image<0,0)
	image = Image.fromarray(np.uint8(image))
	if outPath is None:
		imagePath = imagePath.split('.')
		image.save(imagePath[0]+'Edited'+'.'+imagePath[1])
	else:
		image.save(outPath)


def phaseModify(ftar, l, k):
	height, width = ftar.shape
	for h in range(0,height):
		for w in range(0,width):
			p = complex(0,((-2 * w * k * cmath.pi) / width) + ((-2 * h * l * cmath.pi) / height) + (k + l) * cmath.pi)
			ftar[h,w] *= cmath.exp(p)
	return ftar


def detectEdge(ftar, degree, W, D):
	height, width = ftar.shape
	degree += 90
	angle1 = (degree + W) * (np.pi/180)
	angle2 = (degree - W) * (np.pi/180)
	dc = ftar[height//2,width//2]

	if degree != 90:
		for h in range(0,height):
			for w in range(0,width):
				if ((h < math.tan(angle1) * (w - width / 2) + height / 2 or h > math.tan(angle2) * (w - width / 2) + height / 2 or math.sqrt((w - width//2) * (w - width//2) + (h - height//2) * (h - height//2)) < D) and (h < math.tan(angle2) * (w - width / 2) + height / 2 or h > math.tan(angle1) * (w - width / 2) + height / 2 or math.sqrt((w - width//2) * (w - width//2) + (h - height//2) * (h - height//2)) < D)):
					ftar[h,w] = 0
	else:
		for h in range(0,height):
			for w in range(0,width):
				if ((w < 1/math.tan(angle1) * (h - height / 2) + width / 2 or w > 1/math.tan(angle2) * (h - height / 2) + width / 2 or math.sqrt((w - width//2) * (w - width//2) + (h - height//2) * (h - height//2)) < D) and (w < 1/math.tan(angle2) * (h - height / 2) + width / 2 or w > 1/math.tan(angle1) * (h - height / 2) + width / 2 or math.sqrt((w - width//2) * (w - width//2) + (h - height//2) * (h - height//2)) < D)):
					ftar[h,w] = 0
	ftar[height//2,width//2] = dc
	return ftar


def bandPass(ftar, center, width):
	n = len(ftar)
	m = len(ftar[0])
	lowCut = center - width/2
	highCut = center + width/2
	dc = ftar[n//2,m//2]
	for i in range(0,n):
		for j in range(0,m):
			distance = math.sqrt((i - n//2) * (i - n//2) + (j - m//2) * (j - m//2))
			if distance <= lowCut or distance >= highCut:
				ftar[i,j] = 0
	ftar[n//2,m//2] = dc
	return ftar


def bandCut(ftar, center, width):
	n = len(ftar)
	m = len(ftar[0])
	lowCut = center - width/2
	highCut = center + width/2
	dc = ftar[n//2,m//2]
	for i in range(0,n):
		for j in range(0,m):
			distance = math.sqrt((i - n//2) * (i - n//2) + (j - m//2) * (j - m//2))
			if distance >= lowCut and distance <= highCut:
				ftar[i,j] = 0
	ftar[n//2,m//2] = dc
	return ftar

def lowPass(ftar, cutoff):
	n = len(ftar)
	m = len(ftar[0])
	for i in range(0,n):
		for j in range(0,m):
			if math.sqrt((i - n//2) * (i - n//2) + (j - m//2) * (j - m//2)) >= cutoff:
				ftar[i,j] = 0
	return ftar


def highPass(ftar, cutoff):
	#DO NOT SET DC TO 0
	n = len(ftar)
	m = len(ftar[0])
	dc = ftar[n//2,m//2]
	for i in range(0,n):
		for j in range(0,m):
			if math.sqrt((i - n//2) * (i - n//2) + (j - m//2) * (j - m//2)) <= cutoff:
				ftar[i,j] = 0
	ftar[n//2,m//2] = dc
	return ftar


def performIfft(fftar, outPath):
	saveSpectrum(fftar, outPath)
	savePhase(fftar, outPath)
	n = len(fftar)
	m = len(fftar[0])

	tmp = np.copy(fftar[:n//2,:m//2])
	fftar[:n//2,:m//2] = fftar[n//2:n,m//2:m]
	fftar[n//2:n,m//2:m] = tmp
	tmp = np.copy(fftar[:n//2,m//2:m])
	fftar[:n//2,m//2:m] = fftar[n//2:n,:m//2]
	fftar[n//2:n,:m//2] = tmp

	ifftar = np.zeros((n,m), dtype=complex)
	for i in range(0,n):
		ifftar[i] = ifft(fftar[i])
	for j in range(0,m):
		ifftar[:,j] = ifft(ifftar[:,j])
	ifftar = ifftar/math.sqrt(n*m)

	return np.abs(ifftar)


def ifft(x):
	n = len(x)
	if n <= 1:
		return x
	even = ifft(x[0::2])
	odd = ifft(x[1::2])
	exponents = np.exp((2j * np.pi * np.arange(0,n//2,1))/n)
	T = np.multiply(exponents,odd)
	return np.concatenate((even+T,even-T))


def performFft(image):
	time = perf_counter()
	n = len(image)
	m = len(image[0])
	fftar = np.zeros((n,m), dtype=complex)

	for i in range(0,n):
		fftar[i] = fft(image[i])
	for j in range(0,m):
		fftar[:,j] = fft(fftar[:,j])
	fftar = fftar/math.sqrt(n*m)

	tmp = np.copy(fftar[:n//2,:m//2])
	fftar[:n//2,:m//2] = fftar[n//2:n,m//2:m]
	fftar[n//2:n,m//2:m] = tmp
	tmp = np.copy(fftar[:n//2,m//2:m])
	fftar[:n//2,m//2:m] = fftar[n//2:n,:m//2]
	fftar[n//2:n,:m//2] = tmp
	print('Time elapsed:' + str(perf_counter() - time))

	return fftar

def fft(x):
	n = len(x)
	if n <= 1:
		return x
	even = fft(x[0::2])
	odd = fft(x[1::2])
	exponents = np.exp((-2j * np.pi * np.arange(0,n//2,1)/n))
	T = np.multiply(exponents,odd)
	return np.concatenate((np.add(even,T),np.subtract(even,T)))


def dft(image):
	n = len(image)
	m = len(image[0])
	scaling = 1/math.sqrt(n*m)
	dft = np.zeros((n,m), dtype=complex)
	for i in range(0,n):
		for j in range(0,m):
			sum = 0
			for k in range(0,n):
				for l in range(0,m):
					e = cmath.exp(-2j * np.pi * (((i * k) / n) + (j * l) / m))
					sum += scaling * image[k,l] * e
			dft[i,j] = sum

	tmp = np.copy(dft[:n//2,:m//2])
	dft[:n//2,:m//2] = dft[n//2:n,m//2:m]
	dft[n//2:n,m//2:m] = tmp
	tmp = np.copy(dft[:n//2,m//2:m])
	dft[:n//2,m//2:m] = dft[n//2:n,:m//2]
	dft[n//2:n,:m//2] = tmp

	return dft

def ift(dft):
	saveSpectrum(dft, outPath)
	savePhase(dft, outPath)
	n = len(dft)
	m = len(dft[0])

	tmp = np.copy(dft[:n//2,:m//2])
	dft[:n//2,:m//2] = dft[n//2:n,m//2:m]
	dft[n//2:n,m//2:m] = tmp
	tmp = np.copy(dft[:n//2,m//2:m])
	dft[:n//2,m//2:m] = dft[n//2:n,:m//2]
	dft[n//2:n,:m//2] = tmp

	scaling = 1/math.sqrt(n*m)
	ift = np.zeros((n,m), dtype=complex)
	for i in range(0,n):
		for j in range(0,m):
			sum = 0
			for k in range(0,n):
				for l in range(0,m):
					e = cmath.exp(2j * np.pi * (((i * k) / n) + (j * l) / m))
					sum += scaling * dft[k,l] * e
			ift[i,j] = sum

	return np.abs(ift)

def saveSpectrum(dft, outPath):
	dftImg = np.zeros_like(dft, dtype=float)
	np.putmask(dftImg,dft,np.log(np.abs(dft)+0.001)*50)
	np.putmask(dftImg,dftImg>255,255)
	np.putmask(dftImg,dftImg<0,0)
	dftImg = Image.fromarray(np.uint8(dftImg))
	outPath = outPath.split('.')
	dftImg.save(outPath[0]+'Spectrum'+'.'+outPath[1])

def savePhase(dft, outPath):
	dftImg = np.zeros_like(dft, dtype=float)
	np.putmask(dftImg,dft,np.angle(dft)*100)
	np.putmask(dftImg,dftImg>255,255)
	np.putmask(dftImg,dftImg<0,0)
	dftImg = Image.fromarray(np.uint8(dftImg))
	outPath = outPath.split('.')
	dftImg.save(outPath[0]+'Phase'+'.'+outPath[1])

def m6(image):
	time = perf_counter()
	np.putmask(image,image==1,255)
	out = np.copy(image)
	backmasks = [[(0,0),(1,-1),(1,0),(1,1)], [(0,-1),(0,0),(1,-1),(1,0)], [(-1,-1),(0,-1),(0,0),(1,-1)], [(-1,-1),(-1,0),(0,-1),(0,0)], [(-1,-1),(-1,0),(-1,1),(0,0)], [(-1,0),(-1,1),(0,0),(0,1)], [(-1,1),(0,0),(0,1),(1,1)], [(0,0),(0,1),(1,0),(1,1)] ]
	foremasks = [[(-1,-1),(-1,0),(-1,1)], [(-1,0),(-1,1),(0,1)], [(-1,1),(0,1),(1,1)], [(0,1),(1,0),(1,1)], [(1,-1),(1,0),(1,1)], [(0,-1),(1,-1),(1,0)], [(-1,-1),(0,-1),(1,-1)], [(-1,-1),(-1,0),(0,-1)] ]
	for i in range(0,len(foremasks)):
		while True:
			tmp = hitmiss(out,foremasks[i], backmasks[i])
			oldOut = np.copy(out)
			np.putmask(out,tmp == 0, 0)
			if (np.array_equal(out,oldOut)):
				break
	print('Time elapsed:' + str(perf_counter() - time))
	return out

def hitmiss(image, foremask, backmask):
	np.putmask(image,image==1,255)
	out = np.copy(image)
	out.fill(255)
	h = len(image)-1
	w = len(image[0])-1
	if backmask:
		for i in range(1,h):
			for j in range(1,w):
				out[i,j] = 0 if (max(image[i+a,j+b] for a,b in foremask) == 0 
						and min(image[i+a,j+b] for a,b in backmask) == 255) else 255
	else:
		for i in range(1,h):
			for j in range(1,w):
				out[i,j] = 0 if (max(image[i+a,j+b] for a,b in foremask) == 0) else 255
	return out

def erode(image):
	np.putmask(image,image==1,255)
	out = np.copy(image)
	h = len(image)-1
	w = len(image[0])-1
	#window = [(-1,0),(1,0),(0,0),(0,-1),(0,1)]
	window = [(i,j) 
		   for i in range(-1,2) 
		   for j in range(-1,2)]
	#left and right edges
	for i in range(1,h):
		out[i,0] = max(image[i,0], image[i-1,0], image[i,1], image[i+1,0])
		out[i,w] = max(image[i,w], image[i-1,w], image[i,w-1], image[i+1,w])
	#top and bottom edges
	for i in range(1,w):
		out[0,i] = max(image[0,i],image[1,i], image[0,i-1], image[0,i+1])
		out[h,i] = max(image[h,i], image[h,i], image[h,i-1], image[h,i+1]) 
	#corners
	out[h,0] = max(image[h,1], image[h,0], image[h-1,0])
	out[h,w] = max(image[h,w], image[h-1,w], image[h,w-1]) 
	out[0,0] = max(image[0,0], image[0,1], image[1,0])
	out[0,w] = max(image[0,w], image[1,w], image[0,w-1]) 
	#rest
	for i in range(1,h):
		for j in range(1,w):
			out[i,j] = max(image[i+a,j+b] for a,b in window)
	return out

def dilate(image):
	np.putmask(image,image==1,255)
	out = np.copy(image)
	h = len(image)-1
	w = len(image[0])-1

	#window = [(-1,0),(1,0),(0,0),(0,-1),(0,1)]
	window = [(i,j) 
		   for i in range(-1,2) 
		   for j in range(-1,2)]
	#left and right edges
	for i in range(1,h):
		out[i,0] = min(image[i,0], image[i-1,0], image[i,1], image[i+1,0])
		out[i,w] = min(image[i,w], image[i-1,w], image[i,w-1], image[i+1,w])
	#top and bottom edges
	for i in range(1,w):
		out[0,i] = min(image[0,i],image[1,i], image[0,i-1], image[0,i+1])
		out[h,i] = min(image[h,i], image[h,i], image[h,i-1], image[h,i+1]) 
	#corners
	out[h,0] = min(image[h,1], image[h,0], image[h-1,0])
	out[h,w] = min(image[h,w], image[h-1,w], image[h,w-1]) 
	out[0,0] = min(image[0,0], image[0,1], image[1,0])
	out[0,w] = min(image[0,w], image[1,w], image[0,w-1]) 
	#rest
	for i in range(1,h):
		for j in range(1,w):
			out[i,j] = min(image[i+a,j+b] for a,b in window)
	return out

def growRegion(image, seed):
	time = perf_counter()
	diff = int(seed[2])
	seed = (int(seed[1]),int(seed[0]))
	seedNo = 1
	out = np.zeros_like(image)
	lenX = len(image)
	lenY = len(image[0])
	window = [(-1,0),(1,0),(0,-1),(0,1)]
	explored = []
	neighbors = []
	for i in range(0,seedNo):
		for a,b in window:
			if (min(seed[0]+a,seed[1]+b) < 0 or
		   lenX <= seed[0]+a or
		   lenY <= seed[1]+b):
				continue
			else:
				neighbors.append((seed[0]+a,seed[1]+b))
		out[seed] = 255
		explored.append(seed)
		while len(neighbors) > 1:
			pixel = neighbors.pop()
			if (abs(image[pixel]-image[seed])<=diff):
				for a,b in window:
					if (min(pixel[0]+a,pixel[1]+b) < 0 or
					lenX <= pixel[0]+a or
					lenY <= pixel[1]+b or
					(pixel[0]+a,pixel[1]+b) in explored):
						continue
					else:
						neighbors.append((pixel[0]+a,pixel[1]+b))
					out[pixel] = 255
			explored.append(pixel)
      
	print('Time elapsed:' + str(perf_counter() - time))
	return out
		
def makeHistogram(image, isRgb):
	image = np.uint8(image)
	if isRgb:
		histR = np.bincount(np.reshape(image[:,:,0],image[:,:,0].size),None,256)
		histG = np.bincount(np.reshape(image[:,:,1],image[:,:,1].size),None,256)
		histB = np.bincount(np.reshape(image[:,:,2],image[:,:,2].size),None,256)
		hist = np.zeros([256,3])
		for i in range(0,256):
			hist[i] = [histR[i],histG[i],histB[i]]
		histImgR = np.zeros([len(image),256,3])
		histImgG = np.zeros([len(image),256,3])
		histImgB = np.zeros([len(image),256,3])	
		i=0
		compHist = np.copy(hist)
		while (max(compHist.flatten()) > len(image)):
			compHist = compHist//2
		for val in compHist:
			if val[0] > 0:
				histImgR[len(histImgR)-1:int((len(histImgR)-val[0])):-1,i,:] = (255,0,0)
			if val[1] > 0:
				histImgG[len(histImgG)-1:int((len(histImgG)-val[1])):-1,i,:] = (0,255,0)
			if val[2] > 0:
				histImgB[len(histImgB)-1:int((len(histImgB)-val[2])):-1,i,:] = (0,0,255)
			i+=1
		histImg=[histImgR,histImgG,histImgB]
		return histImg,hist
	else:
		hist = np.bincount(image.flatten(),None,256)
		histImg = np.zeros([len(image),256])
		i = 0
		compHist = np.copy(hist)
		while (max(compHist) >= len(image)):
			compHist = compHist//2
		for val in compHist:
			if val > 0:
				histImg[len(histImg)-1:(len(histImg)-val):-1,i] = 255
			i+=1
		return histImg,hist

def hUniform(image,hist):
	gMin = 0
	gMax = 255
	n=len(image)*len(image[0])
	g = np.arange(0,gMax+1)
	f = np.arange(0,gMax+1)
	for i in range(gMin,gMax+1):
		g[i] = gMin+(gMax-gMin)*(1/n)*sum(hist[0:g[i]+1])
	for i in range(0,len(image)):
		for j in range(0,len(image[0])):
			image[i,j] = np.float64(g[np.where(f==image[i,j])[0]])
	return image

def laplace(image,maskNo):
	time = perf_counter()
	output = np.zeros_like(image)
	window = [(i,j) 
		   for i in range(-1,2) 
		   for j in range(-1,2)]
	if maskNo == 1:
		mask = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
	elif maskNo == 2:
		mask = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
	elif maskNo == 3:
		mask = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]])
	else:
		print("Choose a mask from 1-3")
		return image
	try:
		if len(image[0][0] > 1):
			for i in range(0,len(image)):
				for j in range(0,len(image[0])):
					for k in range(0,3):
						windowSum = sum(
							0 if ((min(i+a,j+b) < 0)
								or (len(image) <= i+a)
								or (len(image[0]) <= j+b)
								or (image[i+a][j+b][k] == 0))
							else image[i+a,j+b,k]*mask[a+1,b+1]
							for a,b in window)
						output[i,j,k] = windowSum
	except TypeError:
		for i in range(0,len(image)):
			for j in range(0,len(image[0])):
				windowSum = sum(
						0 if ((min(i+a,j+b) < 0)
							or (len(image) <= i+a)
							or (len(image[0]) <= j+b)
							or (image[i+a,j+b] == 0))
						else image[i+a,j+b]*mask[a+1,b+1]
						for a,b in window)
				output[i,j] = windowSum
	np.putmask(output,output>255,255)
	np.putmask(output,output<0,0)
	print('Time elapsed:' + str(perf_counter() - time))
	return output

def optLaplace(image):
	time = perf_counter()
	output = np.zeros_like(image)
	window = np.array([(-1,0),(0,-1),(0,0),(0,1),(1,0)])
	mask = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
	try:
		if len(image[0][0] > 1):
			for i in range(1,len(image)-1):
				for j in range(1,len(image[0])-1):
					for k in range(0,3):
						windowSum = sum(
							0 if ((min(i+a,j+b) < 0)
								or (len(image) <= i+a)
								or (len(image[0]) <= j+b)
								or (image[i+a][j+b][k] == 0))
							else image[i+a,j+b,k]*mask[a+1,b+1]
							for a,b in window)
						output[i,j,k] = windowSum
	except TypeError:
		for i in range(1,len(image)-1):
			for j in range(1,len(image[0])-1):
				output[i,j] = image[i,j]*4-image[i-1,j]-image[i,j-1]-image[i+1,j]-image[i,j+1]
	output = output
	np.putmask(output,output>255,255)
	np.putmask(output,output<0,0)
	print('Time elapsed:' + str(perf_counter() - time))
	return output

def wallis(image):
	time = perf_counter()
	output = np.zeros_like(image)
	window = np.array([(-1,0),(0,1),(1,0),(0,-1)])
	try:
		if len(image[0][0] > 1):
			for i in range(0,len(image)):
				for j in range(0,len(image[0])):
					for k in range(0,3):
						windowSum = sum(
							0 if ((min(i+a,j+b) < 0)
								or (len(image) <= i+a)
								or (len(image[0]) <= j+b))
							else image[i+a,j+b,k]
							for a,b in window)
						if (windowSum == 0 or image[i,j,k] == 0):
							output[i,j,k] = 1
						else:
							output[i,j,k] = math.log10((image[i,j,k]**4)/windowSum)/4
	except TypeError:
		for i in range(1,len(image)-1):
			for j in range(1,len(image[0])-1):
				windowProduct = image[i-1,j]*image[i,j+1]*image[i+1,j]*image[i,j-1]
				if (windowProduct == 0 or image[i,j] == 0):
					output[i,j] = 1
				else:
					output[i,j] = abs(math.log10((image[i,j]**4)/windowProduct)/4)
	output = output*5000
	np.putmask(output,output>255,255)
	np.putmask(output,output<0,0)
	print('Time elapsed:' + str(perf_counter() - time))
	return output

def getMean(image,hist):
	n=len(image)*len(image[0])
	mean = 1/n*sum((np.arange(0,256)*hist))
	return mean

def getStdDev(image,hist):
	n=len(image)*len(image[0])
	mean = getMean(image,hist)
	stdSum = np.arange(0,256)
	stdSum = sum((stdSum-mean)*(stdSum-mean)*hist)
	return np.sqrt(1/n*stdSum)

def horizontalFlip(image):
	return np.flip(image,1)

def verticalFlip(image):
	return np.flip(image,0)

def diagonalFlip(image):
	image = horizontalFlip(image)
	image = verticalFlip(image)
	return image

def resize(image,width,height):
	imageH, imageW = image.shape[:2]
	while (height < (imageH//2)) or (width < (imageW//2)):
		image = bilinearInterpolate(image,imageW//2,imageH//2)
		imageH, imageW = image.shape[:2]
	image = bilinearInterpolate(image,width,height)
	return image

def bilinearInterpolate(image,width,height):
	imageH, imageW = image.shape[:2]
	try: 
		if len(image[0][0] > 1):
			resized = np.empty([height,width,3])
	except TypeError:
		resized = np.empty([height,width])

	xRatio = float(imageW-1)/(width-1) if width > 1 else 0
	yRatio = float(imageH-1)/(height-1) if height > 1 else 0

	for i in range(0,height):
		for j in range(0,width):
			xl,yl = int(np.floor(xRatio*j)), int(np.floor(yRatio*i))
			xh,yh = int(np.floor(xRatio*j)), int(np.floor(yRatio*i))

			xWeight = (xRatio * j) - xl
			yWeight = (yRatio * i) - yl

			a = image[yl][xl]
			b = image[yl][xh]
			c = image[yh][xl]
			d = image[yh][xh]

			resized[i][j] = a * (1-xWeight) * (1-yWeight) + \
				b * xWeight * (1-yWeight) + \
				c * yWeight * (1-xWeight) + \
				d * xWeight * yWeight

	return resized
	
def changeBright(image, val):
	if val > 0 and val <= 255:
		upLim = 255 - val
		loLim = 0
	elif val < 0 and val >= -255:
		upLim = 255
		loLim = 0 - val
	else:
		return image
	np.putmask(image,image>upLim,255)
	np.putmask(image,image<loLim,0)
	np.putmask(image,np.logical_and(image>=loLim,image<=upLim),image+val)
	return image

def changeContrast(image,val):
	if val < 0 or val == 1:
		return image
	else:
		np.putmask(image,image,(image-128)*val+128)
		np.putmask(image,image>255,255)
		np.putmask(image,image<0,0)
	return image

def makeNegative(image):
	np.putmask(image,image,255-image)
	return image

def filterHarmonic(image,filterSize):
	if filterSize<2:
		return image
	time = perf_counter()
	output = np.zeros_like(image)
	indexer = filterSize // 2
	window = [(i,j) 
		   for i in range(-indexer,filterSize-indexer) 
		   for j in range(-indexer,filterSize-indexer)]
	windowSum = 0
	try:
		if len(image[0][0] > 1):
			rgb = True
	except TypeError:
		rgb = False
	for i in range(0,len(image)):
		for j in range(0,len(image[0])):
			if rgb:
				for k in range(0,3):
					windowSum = sum(
						0 if ((min(i+a,j+b) < 0)
							or (len(image) <= i+a)
							or (len(image[0]) <= j+b)
							or (image[i+a][j+b][k] == 0))
						else 1/image[i+a][j+b][k]
						for a,b in window)
					if windowSum == 0:
						output[i][j][k] = 0
					else: 
						if (filterSize*filterSize)/windowSum >= 255:
							output[i][j][k] = 255
						else:
							output[i][j][k] = (filterSize*filterSize)/windowSum
			else:
				windowSum = sum(
						0 if ((min(i+a,j+b) < 0)
							or (len(image) <= i+a)
							or (len(image[0]) <= j+b)
							or (image[i+a][j+b] == 0))
						else 1/image[i+a][j+b]
						for a,b in window)
				if windowSum == 0:
					output[i][j] = 0
				else: 
					if (filterSize*filterSize)/windowSum >= 255:
						output[i][j] = 255
					else:
						output[i][j] = (filterSize*filterSize)/windowSum
	print('Time elapsed:' + str(perf_counter() - time))
	return output

def filterMedian(image, filterSize):
	if filterSize<2:
		return image
	time = perf_counter()
	output = np.zeros_like(image)
	indexer = filterSize // 2
	window = [(i,j) 
		   for i in range(-indexer,filterSize-indexer) 
		   for j in range(-indexer,filterSize-indexer)]
	index = len(window) // 2
	try:
		if len(image[0][0] > 1):
			rgb = True
	except TypeError:
		rgb = False
	for i in range(0,len(image)):
		for j in range(0,len(image[0])):
			if rgb:
				for k in range(0,3):
					output[i][j][k] = sorted(
					0 if ((min(i+a,j+b) < 0)
						or (len(image) <= i+a)
						or (len(image[0]) <= j+b)
						)
					else image[i+a][j+b][k]
					for a,b in window)[index]
			else:
				output[i][j] = sorted(
					0 if ((min(i+a,j+b) < 0)
						or (len(image) <= i+a)
						or (len(image[0]) <= j+b)
						)
					else image[i+a][j+b]
					for a,b in window)[index]
	print('Time elapsed:' + str(perf_counter() - time))
	return output
			
def getMSE(original, compImg):
	sum=0
	for i in range(0,len(original)):
		for j in range(0,len(original[0])):
			sum+=(original[i][j]-compImg[i][j])*(original[i][j]-compImg[i][j])
	return (1/(len(original)*len(original[0])))*sum

def getPeakMSE(original,compImg):
	try:
		if (len(original[0][0] > 1)):
			sum=0
			peak = np.zeros(3,np.float64)
			for i in range(0,len(original)):
				for j in range(0,len(original[0])):
					sum+=(original[i][j]-compImg[i][j])*(original[i][j]-compImg[i][j])
					peak[0] = max(peak[0],original[i][j][0])
					peak[1] = max(peak[1],original[i][j][1])
					peak[2] = max(peak[2],original[i][j][2])
			mse = (1/(len(original)*len(original[0])))*sum
			peak *= peak
	except TypeError:
		sum=0
		peak = 0
		for i in range(0,len(original)):
			for j in range(0,len(original[0])):
				sum+=(original[i][j]-compImg[i][j])*(original[i][j]-compImg[i][j])
				peak = max(peak,original[i][j])
		mse = (1/(len(original)*len(original[0])))*sum
		peak *= peak
	return mse/peak
	
def getMD(original, compImg):
	try:
		if (len(original[0][0] > 1)):
			md = np.zeros(3,np.float64)
			for i in range(0,len(original)):
				for j in range(0,len(original[0])):
					md[0] = max(md[0],original[i][j][0] - compImg[i][j][0])
					md[1] = max(md[1],original[i][j][1] - compImg[i][j][1])
					md[2] = max(md[2],original[i][j][2] - compImg[i][j][2])
	except TypeError:
		md = 0
		for i in range(0,len(original)):
				for j in range(0,len(original[0])):
					md = max(md, original[i][j] - compImg[i][j])
	return md

def getSNR(original, compImg):
	sumOrig = 0
	sumDiff = 0
	for i in range(0,len(original)):
		for j in range(0,len(original[0])):
			sumOrig += original[i][j]*original[i][j]
			sumDiff += (original[i][j]-compImg[i][j])*(original[i][j]-compImg[i][j])
	return 10*np.log10(sumOrig/sumDiff)

def getPeakSNR(original,compImg):
	sumDiff = 0
	try:
		if len(original[0][0] > 1):
			peak = [0,0,0]
			for i in range(0,len(original)):
				for j in range(0,len(original[0])):
					peak[0] = max(peak[0],original[i][j][0])
					peak[1] = max(peak[1],original[i][j][1])
					peak[2] = max(peak[2],original[i][j][2])
					sumDiff += (original[i][j]-compImg[i][j])*(original[i][j]-compImg[i][j])
	except TypeError:
		peak = 0
		for i in range(0,len(original)):
			for j in range(0,len(original[0])):
				peak = max(peak,original[i][j])
				sumDiff += (original[i][j]-compImg[i][j])*(original[i][j]-compImg[i][j])
	return 10*np.log10((len(original)*len(original[0])*peak*peak)/sumDiff)

if __name__ == "__main__":
	main(sys.argv[1:])
  ```
