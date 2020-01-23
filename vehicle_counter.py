
import cv2
import numpy as np
import matplotlib.pyplot as plt

def VehicleCounter(video0,ini,fin):
    video = cv2.VideoCapture(video0)
    ###Estimacao da imagem de fundo
    imagemFundoEstimada = estimacaoImagemFundo(video)

    nFrame = 1
    numVeiculosDetetados = 0
    contornosFrameAnterior = None
    detetouEsquerda = False
    detetouMeio = False
    detetouDireita = False

    while(True):
        ret, frame = video.read()
        if(nFrame >= ini and nFrame <= fin):
            ###Detecao de pixeis ativos, operadores e detecao regioes
            frame, contornosAtuais = deteccao(frame,imagemFundoEstimada, nFrame)
            #cv2.imwrite('frame_{}.jpg'.format(nFrame),frame)
            ###Classificacao
            numVeiculosFrame = classificacao(frame, contornosAtuais, contornosFrameAnterior)
            contornosFrameAnterior = contornosAtuais
            numVeiculosDetetados += numVeiculosFrame

        ###Visualizacao Resultados
        cv2.putText(frame, "Numero Veiculos Detetados: {} ".format(numVeiculosDetetados), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 204, 204), 2)
        cv2.imshow('Deteccao e contagem de veiculos', frame)
        #cv2.imwrite('frame_class{}.jpg'.format(nFrame),frame)

        if cv2.waitKey(30) & 0xff == ord('q'):
            break

        nFrame +=1

    video.release()
    cv2.destroyAllWindows()

###############################################################################
########################## METODOS AUXILIARES #################################

def estimacaoImagemFundo(cap):
    count = 0
    imgs = []
    while(count<151):#usar valor impar para mediana
        ret, frame = cap.read()
        #final = cv2.medianBlur(frame, 3) #aplica filtro mediana a video
        imgs.append(frame[:,:,2]) #plano red
        count+=1

    med = calculoMediana(imgs)
    #cv2.imwrite('mediana.jpg',med)
    return med

##Calculo da Mediana Temporal
def calculoMediana(imgs):
    nLinhas = range(len(imgs[0][0]))
    nColunas = range(len(imgs[0]))
    mediana = []
    for j in nColunas:
        linha = []
        for i in nLinhas:
            pixel = []
            for img in imgs:
                pixel.append(img[j][i])
            pixel.sort()
            linha.append(pixel[int(len(pixel)/2)])
        mediana.append(linha)

    mediana = np.asarray(mediana) #para poder ver no imshow
    return mediana

def compararPlanos(img):
    plt.figure(1)

    # Red
    plt.subplot(231)
    plt.imshow(img[:,:,2],'gray')
    plt.title('R')

    # Green
    plt.subplot(232)
    plt.imshow(img[:,:,1],'gray')
    plt.title('G')

    # Blue
    plt.subplot(233)
    plt.imshow(img[:,:,0],'gray')
    plt.title('B')

    # Hist Red
    plt.subplot(234)
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(hist)
    plt.title('R Hist')

    # Hist Green
    plt.subplot(235)
    hist = cv2.calcHist([img],[1],None,[256],[0,256])
    plt.plot(hist)
    plt.title('G Hist')

    # Hist Blue
    plt.subplot(236)
    hist = cv2.calcHist([img],[2],None,[256],[0,256])
    plt.plot(hist)
    plt.title('B Hist')

    plt.tight_layout()
    plt.show()

def binarizacao(img):
    ret,thresh = cv2.threshold(img,40,255,cv2.THRESH_BINARY)
    return thresh

def melhoramento(img):
    #Elementos estruturantes
    eKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    eKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,15))
    eKernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,20))

    #Erosao + dilatacao = abertura
    #Erosao
    melhoramento = cv2.erode(img, eKernel)
    #Dilatacao
    melhoramento = cv2.dilate(melhoramento, eKernel2)
    #Fecho (dilatacao + erosao)
    melhoramento = cv2.morphologyEx(melhoramento, cv2.MORPH_CLOSE, eKernel3)
    return melhoramento

def centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return (cx, cy)

def deteccao(frame,imgFundo, nFrame):
    ###Histograma de comparacao dos planos rgb
    #compararPlanos(frame)
    ###Plano Red [:,:,2]
    frameRed = frame[:,:,2]
    ### Regiao ativa
    frameRegiaoAtiva = frameRed[150:,:].copy()
    imgFundoRegiaoAtiva = imgFundo[150:,:].copy()
    ### calcular diferenca entre imagem de fundo encontrada e frame corrente
    frameDelta = cv2.absdiff(imgFundoRegiaoAtiva, frameRegiaoAtiva)
    ### binarizar imagem
    frameBinarizada = binarizacao(frameDelta)
    #cv2.imwrite('binarizacao{}.jpg'.format(nFrame),frameBinarizada)
    ### dilatar a imagem binarizada para preencher pixeis ativos
    frameMelhorada = melhoramento(frameBinarizada)
    #cv2.imwrite('operadoresMorfologicos{}.jpg'.format(nFrame),frameMelhorada)
    ### encontrar contornos na imagem binarizada
    (cnts, _) = cv2.findContours(frameMelhorada.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    ### Percorrer contornos
    frame = frame.copy()
    for c in cnts:
        # calcula a bounding box do contorno
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y+150), (x + w, y+150 + h), (0, 255, 0), 2)
        cx, cy = centroid(x, y, w, h)
        cv2.circle(frame, (cx, cy+150), 1, (0, 255, 0), 2)

    return frame, cnts

def classificacao(frame, contornosAtuais, contornosFrameAnterior):
    numVeiculos = 0
    #print '--------'
    for c in contornosAtuais:
        x, y, w, h = cv2.boundingRect(c)
        #centroide
        moment = cv2.moments(c)
        cx = int(moment['m10']/moment['m00'])
        cy = int(moment['m01']/moment['m00'])
        #print cx, cy

        for cAnt in contornosFrameAnterior:
            xAnt, yAnt, wAnt, hAnt = cv2.boundingRect(cAnt)
            #centroide
            momentAnt = cv2.moments(cAnt)
            cxAnt = int(momentAnt['m10']/momentAnt['m00'])
            cyAnt = int(momentAnt['m01']/momentAnt['m00'])
            #print cx, cy
            if abs(cx-cxAnt)<=5.0:
                #print cyAnt, cy
                if cyAnt > 45.0 and cy <= 45.0:
                    numVeiculos +=1
    return numVeiculos


###############################################################################

if __name__ == '__main__':
    #video 320x240
    video0 = 'Feijo02Setembro2004.mp4'
    VehicleCounter(video0,1,5000)
