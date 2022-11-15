# Convolutional Neural Network - CNN

Trata-se de uma Rede Neural Convolucional (Convolutional Neural Network - CNN) para análise de radiografias. Foram apresentados dois datasets de imagens à CNN:

# - Radiografias pulmonares de pessoas sem coronavírus (mas que apresentam outras doenças);
# - Radiografias pulmonares de pessoas com coronavírus. 

Antes de submeter as imagens à rede neural, as radiografias passaram pelo tratamento de convolução e max pooling (entenda melhor nos links abaixo, que, aliás, me guiaram na execução do código). Também removi imagens que considerei não adequadas do conjunto (qualidade ruim, lateralizadas, viradas...), caso contrário, teria que fazer código para o tratamento dessas imagens e novas classes.

Após o treinamento, foram fornecidas à CNN imagens que foram separadas para teste, ou seja, que não estavam no conjunto de radiografias usadas para o treinamento. Isso significa que a CNN avaliou, ao término, imagens que ela nunca tinha processado. 

Minha playlist de programação:
https://youtube.com/playlist?list=PLH6D3VQZKE164wCyCtynGGtZBr4wpbXUW

Links úteis:
https://developers.google.com/codelabs/tensorflow-2-computervision#0
https://developers.google.com/codelabs/tensorflow-3-convolutions#0
https://developers.google.com/codelabs/tensorflow-5-compleximages#0

Assista esse vídeo:
https://youtu.be/fbxVrARF0a8

