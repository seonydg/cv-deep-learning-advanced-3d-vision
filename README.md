# Computer Vision Deep Learning Advanced: 3D Vision

1. Point Transformer


---
# 1. Point Transformer
# Point Transformer(ICCV, 2021)

## PointNet

Point Transformer에 앞서 PointNet를 먼저 살펴보자면, 구조는 아래 그림과 같다.
먼저 classification Network를 거치는데, input cloud points가 들어오면 transformation을 거치고 mlp shared를 이용하여 feature를 학습한 다음 symmetric function 중 하나인 max pool을 이용하여 global feature를 추출한다. 
그리고 segmentation을 위해 학습된 global feature를 segmentation Network를 거쳐서 output scores를 출력하게 된다.

![](https://velog.velcdn.com/images/seonydg/post/ec5d1d95-90bb-4ff9-94f7-d75f83636de3/image.png)

2D에서는 로컬리티를 이용하는 반면, PointNet은 로컬리티를 사용하지 않기 때문에 3D를 처리하는데 비효율적이다. 새로 보지 못한 scene의 설정에 대해서는 일반화하기 어려운 점이 있다.

point clouds에서 locality는 3가지를 말한다. 
먼저 k-nearest neighbor search의 방법으로 가장 가까운 k개의 point를 설정하는 것이다. k개의 point를 찾기 위해서 거리를 계산하고 크기 순으로 정렬한 다음 k개를 뽑는 형태이다.
다음으로는 radius search로, 일정한 길이인 반지름을 설정한 다음, 설정한 반지름에 해당하는 원 안의 point를 정하는 방식이다. 이 방식은 모든 point 사이의 거리를 계산할 때 효율적으로 계산할 수 있다는 장점이 있다. r보다 큰 point는 계산할 필요가 없기 때문이다.
마지막으로 앞의 2가지 방식을 합친 것으로 hybrid search 방식이 있다. r(반지름) 안에 들어가는 point를 찾은 다음 정해진 k의 개수보다 많다면 더 가까이 있는 k개의 point를 설정하는 방식이다.

![](https://velog.velcdn.com/images/seonydg/post/1614e284-7f9b-46f0-95b4-847f1050167d/image.png)

이러한 로컬리티를 이용해서 PointNet에 로컬리티 특성을 추가한 것이 아래 그림의 PointNet++이다.
N은 point clouds의 크기를, d는 point의 dimension으로 3D points를 사용하기에 d는 3을, PointNet++에서는 local neighbor를 정의할 때 hybrid search 방식을 선택하는데 여기에서 개수 k에서 개수를 K로, C는 각 feature map에서 channel dimension을 말한다.

![](https://velog.velcdn.com/images/seonydg/post/494ba1d1-2863-4d36-a9c3-6e5e876b67b7/image.png)

input cloud points가 들어오면, conv2d에서 feature map을 줄여나가듯이 3D에서도 포인트를 샘플링해서 feature map을 줄인다. 그렇게 레졸루션이 차이가 나기 때문에 hierarchical point set feature learning이 가능하게 된다. 그리고 2D의  U-Net에서 인코더-디코더처럼, PointNet++ 구조에서는 feature hierarchical가 존재하기에 segmentation 부분에서 디코더를 수행할 수 있게 된다.

![](https://velog.velcdn.com/images/seonydg/post/94a7230a-6111-43fa-9089-4c2f7a22cabe/image.png)

샘플링은, xyz 3차원 coordinates of input points에 해당하는 feature를 가지는 N크기의 'd' dimension이 Farthest Point Sampling을 통해서 N보다 작은 N1 크기의 point를 'sampling'한다.
그리고 샘플링 된 point를 Ball query(hybrid search)을 이용하여 'grouping'을 진행한다.
그러면 아래와 같이 3차원의 N, K, d+c shape의 3차원 array 형테로 neighbor features를 정의할 수 있게 된다.

![](https://velog.velcdn.com/images/seonydg/post/dd748ea3-ad58-4325-9950-e6ce63a24aad/image.png)

이 후에 PointNet을 적용시키게 되는데, PointNet 아키텍처 그림에서 n*3에서 n에 해당하는 point clouds가 neighbor features로 들어가게 된다.

이런 과정을 거쳐서 인코딩이 된 feature map을 다시 디코더 과정인 Upsampling을 거치게 된다.
인코더 단에서 중간 단계의 레졸루션 point clouds가 어떻게 생겼는지 알고 있기 때문에 해당 구조를 Upsampling 과정에 그대로 사용하게 된다. 그리고 U-Net과 같이 skip connector를 통해서 concatenate하게 되고, convolution을 수행하듯이 pointNet으로 feature를 섞어주는 과정이 추가로 들어간다.



## Point Transformer
**Point Transformer**는 PointNet++과 유사하지만 **k-nearest neighbor search**를 이용하여 neighbor를 정의하게 된다. hybrid search에서 r이 limmit을 설정하기에 장점도 있지만, point가 매우 조밀하거나 반대의 경우 k개의 포인트만 찾게 되어 영역이 많이 달라질 수 있다. 반면 k-nearest neighbor search는 local neighborhood에서 transformer 구조를 수행하는 방식으로 k개의 가장 가까운 neighbor를 보장함으로써 self-attention을 안정적으로 학습하는 효과가 있다.
그리고 self-attention을 통해서 feature를 학습하고 summation을 수행해서 feature learning을 하는 것이 다른 점이다.

Point Transformer도 아래의 그림과 같이 다운 샘플링과 업 샘플링을 하는데 그 방식은 PointNet++ 방식과 비슷하다.

![](https://velog.velcdn.com/images/seonydg/post/a81c22a1-adce-4ed6-a0e4-c56f41f9063d/image.png)


### Local Self-Attention

Local Self-Attention은 중간의 feature에 따라서 conv kennel의 weight이 바뀌는 다이나믹 커널 웨이트를 가지고 있다.
아래의 그림에서 오른쪽처럼 일반적으로 kennel의 크기에 맞게 각 위치를 곱하고 더해서 output을 내뱉는 것과는 달리, 입력된 feature의 가운데 feature에서 **query** feature를 만들고 자기 자신을 포함한 neighborhood 안에 들어가는 features을 통해서 **key**를 만들고, query와 key의 similarities를 계산하여 합이 1이 되는 다이나믹한 kennel weight를 만든다. 그리고 local neighborhood를 **value**로 만들어 **similarity weight**랑 합쳐서 output을 내뱉게 된다.
일반적인 conv layer와 다른 점은 learned weights가 바뀌지 않는 것과 달리, Local Self-Attention은 local neighborhood에 존재하는 key들이 바뀌면 conv kennel과 similarity metrics가 바뀌어서 output도 바뀌게 된다. 

![](https://velog.velcdn.com/images/seonydg/post/237d3078-e21c-46f1-a24d-e7e803aa7ef3/image.png)

similarity는 먼저 query와 keys를 Dot-product를 하고 softmax와 같이 normalization을 진행하고, 다시 value와 Dot-product를 하여 정의하게 된다. query와 keys의 Dot-product한 것을 normalization function을 이용해서 summation을 하면 query와 keys의 similarity의 총합은 '1'이라는 제약 조건이 존재하게 된다. 그리고 values에 weighted sum을 진행하면 가장 similarity가 높은 key에 해당하는 value와 가장 유사하게 output이 출력된다.

![](https://velog.velcdn.com/images/seonydg/post/787829e3-5945-4db3-a27d-17ccaa506946/image.png)

![](https://velog.velcdn.com/images/seonydg/post/3858f5b1-68a1-4ce7-9662-ae16d7e44a9a/image.png)

이런 Local Self-Attention은 기본적으로 keys의 순서가 바뀌어도 output은 변하지 않는다는 특성이 있다. 그래서 문자나 이미지를 처리할 때는 local이 섞이는 것을 방지하기 위해 기존의 transformer과 같이 positional encoding을 더해주게 된다.

![](https://velog.velcdn.com/images/seonydg/post/a0f31d64-ec4e-4c66-9771-1172ae46489b/image.png)

Point Transformer은 kNN search with local(vector) self-attention을 사용한다. 여기에서 델타가 positional encoding이다.

![](https://velog.velcdn.com/images/seonydg/post/acbed9b9-9311-42c9-bb5c-50eade4839a6/image.png)

아래는 ate of the art performance** 결과이다.

![](https://velog.velcdn.com/images/seonydg/post/c0a4068d-458f-4fc2-a1fa-b9cf7b996ea0/image.png)
