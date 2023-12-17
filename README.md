# Computer Vision Deep Learning Advanced: 3D Vision

1. Point Transformer
2. VoteNet
3. SPVNAS


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


---
# 2. VoteNet
# VoteNet(CVPR, 2020)

Indoor Scenes은 기본적으로 데이터 RGBD(RGB + Depth Camera) 스캐닝 방식을 사용한다.
**Indoor 3D Scenes Object Detection**의 input은 RGBD 이미지로 output은 3D 바운딩 박스로 z축 방향의 rotation이 있는 바운딩 박스를 해당하는 class label과 함께 예측하게 된다.

![](https://velog.velcdn.com/images/seonydg/post/baa28be4-6771-4875-a817-17c827c82735/image.png)

VoteNet(Deep Hough Voting for 3D Object Detection in Point Clouds)은 2D Detector를 사용하지 않는다. 그리고 Point Clouds 자체에서 3D Detection을 수행한다. 그리고 RGBD를 사용하지 않고 Depth Image만을 사용하여도, 기존의 RGBD Image를 사용하는 방법들보다 좋은 성능을 냈다는 특징이 있다.


### Hough Voting

먼저 Hough Voting 방식이 무엇인지 잠시 살펴보도록 하자.
Hough Voting은 2D Detection에서 사용되던 방법으로, 아래의 그림과 같이 엣지 부분 등으로 관심가는 포인트들을 찾고 그 포인트의 일정한 크기의 패치를 추출한다. 그리고 패치에서 training에서 존재하는 패치 중 유사한 패치를 찾는다. training에서 각 패치마다 어떤 object center와 연관되어 있는지 알고 있고 training image를 바탕으로 voting을 하게 된다. 

![](https://velog.velcdn.com/images/seonydg/post/90792248-e59d-408a-b1a3-f28d5abcaa2a/image.png)

모든 voting이 옳바르게 voting이 될 수는 없다. 예로 아래의 그림과 같이 산의 엣지 부분의 패치가 '소'의 엉덩이 부분이랑 잘못 매칭이 되는 경우도 발생한다. 

![](https://velog.velcdn.com/images/seonydg/post/6ed85440-c415-4231-9b99-ea0e9fe6a5cc/image.png)

그래서 모든 매칭이 이루어진 다음에는 아래와 같이 voting이 밀집되어 있는 곳만 제대로 매칭이 된 voting라고 판단을 하고 Object Detection에 사용하게 된다. 즉 voting space에서 peak 부분만 사용한다고 볼 수 있다.

![](https://velog.velcdn.com/images/seonydg/post/cb9579ba-1434-4d73-9dd1-2fa6b48874b8/image.png)

그리고 peakㄹㄹ 찾고 난 이후에는 voting에 기여한 패치들이 어디 있는지 다시 back projection하여 찾는다. 그리고 패치들을 바탕으로 아래와 같이 바운딩 박스에 대한 정보를 확인할 수 있게 된다.
이것이 2D Hough Voting의 방법을 이용한 Object detector다.

![](https://velog.velcdn.com/images/seonydg/post/22b534cf-07b9-43ce-a7d5-12e021faf55d/image.png)


## VoteNet 수행 단계

아래의 그림을 참조하여 살펴보자.

**Input**
**VoteNet**은 N개의 input을 Point cloud feature를 추출할 수 있는 backbone을 이용하여 Point feature를 추출한다. 논문에서 사용되는 backbone은 PointNet++을 사용한다.

**Seeds**
추출된 Point feature를 interest points(Seed Point)를 샘플링한다. 샘플링 과정은 Point Transformer에서 사용된 Farthest point sampling(FPS)를 사용한다. 

**Votes**
샘플링을 통해 추출된 M개의 Seed Points에 대하여 Shared MLP를 이용한 voting 과정을 수행한다. voting 과정은 각 seed point에 할당된 object의 center를 에측하는 과정이다. 

**Vote clusters**
Vote 과정에서 빨간점처럼 vote가 있으면 clustering을 통해 voting의 peak 지점을 찾는다. cluster의 개수는 미리 지정하는 k개로 FPS를 이용하여 k개의 cluster center를 먼저 샘플링한다. 

**Output**
그 이후에 radius search를 이용한 grouping을 수행하여 같은 cluster 안에 있는 Point feature를 aggregation하여 마지막 바운딩 박스를 예측하게 된다.

![](https://velog.velcdn.com/images/seonydg/post/45a3a893-bc02-4f9d-9646-3b153c9da255/image.png)

VoteNet은 기본적으로 **2 Stage Object Detection** 형식을 따르고 있기 때문에 마지막 부분에 3D non-maximum suppression(NMS)과정이 필수적으로 들어가게 된다.



**2D Hough Voting**에서는 바운딩 박스를 찾기 위해 voting을 하는데 기여한 패치들을 back projection하는 과정이 필요했던 반면, 3D VoteNet에서는 clustering하여 찾아낸 cluster features를 aggregation하여 Pooling할 것인지 MLP를 통해서 학습을 하게 된다.


## Loss

**Vote Regression Loss**
Seed point들이 object 중심을 잘 예측할 수 있도록 Regression Loss가 있다. 
그 다음은 2 Stage Object Detection 형식을 따르고 있다.

**Classification Loss**
Proposal된 바운딩 박스가 object인지 아닌지 분류하는 Classification Loss가 있다.

**Bounding Box Regression Loss**
예측된 Bounding Box의 크기를 예측하는 Regression Loss가 있다.

**semantic Classification**
예측된 Bounding Box에서 각 semantic class가 무엇인지 분류하는 Classification Loss가 있다.

![](https://velog.velcdn.com/images/seonydg/post/c37206e0-c1f4-4e73-890e-181210dbe384/image.png)

**Vote Regression Loss** 다음의 3가지 Loss는 기존의 2 Stage Object Detection에서 사용한 Loss와 같다.

Vote Regression Loss 좀 더 살펴보자면, Seed point가 object일 때 Loss를 계산한다. 
아래의 계산식에서 보면, s1(seed point) on object는 Seed point가 object 위에 있을 때를 의미하는 것으로 Seed point에만 Loss를 주고 아닐 때에는 Loss를 주지 않는다. 
그러면 object인지 아닌지 판단하는 것은 아래의 그림과 같이, object 중심으로부터 0.3 이내에 있으면 objcet 안에 있다고 보고 0.6보다 밖에 있으면 아니라고 본다.

![](https://velog.velcdn.com/images/seonydg/post/6e58ab39-bfa2-42e7-bbbb-9d06983ccf35/image.png)

실험 결과를 살펴보면. VoteNet의 Input은 Geo metric(Depth)만을 사용하여도 기존의 방법들보다 좋은 성능을 보이고 있다.

![](https://velog.velcdn.com/images/seonydg/post/cd9fb342-c7b2-4427-a21f-8a77310aa5c2/image.png)

아래는 ate of the art performance** 결과이다.

![](https://velog.velcdn.com/images/seonydg/post/c0a4068d-458f-4fc2-a1fa-b9cf7b996ea0/image.png)


---
# 3. SPVNAS
# SPVNAS(EECV, 2020)

## Outdoor Scenes

Outdoor Scenes은 다음과 같은 특징이 있다.
먼저 Light Detection And Ranging(LiDAR) 센서를 이용해서 주로 데이터를 얻는데 아래의 그림과 같이 생겼다. 센서를 통해서 얻어지는 데이터는 찍은 위치에서 멀어질수록 데이터가 희미해지는 특징이 있다. 그리고 찍힌 포인터들이나 공간적 영역이 굉장히 넓다. 그래서 Indoor Scenes에서 사용되던 Point Transformer 혹은 PointNet++와 같은 모델을 바로 적용시키기엔 너무 무겁다.

![](https://velog.velcdn.com/images/seonydg/post/d07fb562-13ba-4684-a41f-1b95a98f94fd/image.png)

이러한 특징으로 Voxel을 사용한다. 복셀은 Volume과 Element의 합성어로 굉장히 넓은 영역을 효과적으로 처리하는데 많이 사용되고 있다.
LiDAR Point의 경우, 기본적으로 **Sparse & Irregular**한 Point Clouds이기에 복셀로 만드는 과정이 필요한데, 그 과정을 **Voxelization**이라고 한다.
일반적으로 복셀은 정육면체를 사용하여 Sparse & Regular하게 만든다.

![](https://velog.velcdn.com/images/seonydg/post/a159b5a1-79cb-483c-a94e-7a1d9f3a923c/image.png)

이렇게 Regular하게 만든 복셀 그리드는 Convolution 연산을 쉽게 할 수 있게 된다.
일반적으로 2D처럼 연산을 하게 되면 빈공간이 너무 많아 필요없는 연산이 많아지게 된다. 그래서 제안된 것이 Submanifold Sparse Convolution이다. Output에서 정보가 있는 위치만 찾아서 Convolution 연산을 진행하는 것이다.
그리고 Output이 Input과 동일한 Sparsity가 필요하지 않은데, 그것을 Generalized Sparse Convolution이라고 한다. Output Sparsity를 기준으로 정보가 있는 복셀이 주변에 있는지 찾는 과정이 더해지게 된다.
이 두가지의 차이점은 정적(Static)인지 다이나믹(Dynamic)인지에 달렸다.

![](https://velog.velcdn.com/images/seonydg/post/3e7a1e8e-307c-42b5-bb52-a7d0204962a3/image.png)


## Sparse Convolution

Sparse Convolution 연산이 어떻게 진행되는지 보자.
아래와 같이 Point Clouds가 입력이 되면 PointNet과 같이 xyz 좌표를 sharded MLP를 통해 Symmetric function을 거쳐 Global feature를 만들게 된다면, LiDAR와 같은 경우에는 Point가 굉장이 많은 데이터를 처리하기에 너무 무겁다.

![](https://velog.velcdn.com/images/seonydg/post/7541a054-db6a-4606-b4e7-c6c4c5bd9750/image.png)

그래서 먼저, Sparse Convolution을 사용하면 아래와 같이, voxelization을 통해서 Sparse voxel grid로 만든다.

![](https://velog.velcdn.com/images/seonydg/post/4d57111b-db43-4941-8d6f-c27af3bb5c05/image.png)

그 다음, 각 복셀에서 주변 point를 찾고 Colvolution을 수행하게 된다.
아래의 그림의 초록색 위치에서 데이터가 있는 복셀이 있는지 먼저 찾고, 미리 정의되어 있는 kennel shape에 따라서 각각의 위치에 weights과 bias와 연산이 이루어진다.
기존의 Convolution과는 달리 Sparse Convolution은 데이터가 있는 부분에서만 연산을 수행하게 된다. 그리고 summation을 통해서 Output을 내뱉는다.
그 다음 Output을 활성함수를 통과하게 되면 최종적인 Output feature가 수행되게 된다.

![](https://velog.velcdn.com/images/seonydg/post/f1c5d6e9-fc88-4544-844b-c72c5fa5534e/image.png)


## Efficient Neighbor Search with Hash Table

hash table을 사용하여 데이터가 있는 위치를 빠르게 찾는다.
아래의 그림과 같이 Input data가 있다면 Quantized를 하게 되면 인티져 값들로 각각의 위치를 표현할 수 있게 되고, 이 인티져값을 이용하여 Hash table을 만들 수 있다. 인티저는 하나의 Hash key가 되고 그 key에 해당하는 Index를 저장한다.

![](https://velog.velcdn.com/images/seonydg/post/9ee5a27c-4fa7-4bff-9178-3da04c77c311/image.png)

예로 아래와 같이 Quantized data에서 (5, 5, 5)라는 복셀 근처에 존재하는 복셀은 무엇인지 kennel shape에 따라서 후보군 Query를 만들 수 있고 이 Query가 Hash table에 있는지 없는지 확인하여 찾게 된다.

![](https://velog.velcdn.com/images/seonydg/post/41973c4a-c7b6-4005-972e-47df1ea6c4d5/image.png)

Voxelization을 수행하게 되면 아무래도 detail이 없어지는 단점을 가지지만 수행 속도가 빨라진다는 장점을 가지게 된다. Image로 보자면 고화질에서 저화질로의 변화가 예라고 볼 수 있다. 그래서 정확도 측면에서는 다소 떨어지는 모습을 가지고 있다.
아래의 표에서 그 차이점을 표현하였는데, Point-based와 Voxel-based의 단점을 보완한 새로운 Operation이 바로 SPVNAS다. SPVNAS는 복셀과 Input-point cloud를 모두 사용하는데, 메모리는 사용량은 올라가지만 주변 데이터를 찾을 때 복셀을 이용하기 때문에 빠르게 찾을 수 있고, 정확도 측면에서도 Voxelization을 통해 잃게 되는 위치의 정확도 측면을 Point 기반의 방법으로 보완하여 복셀 기반보다는 좋은 성능을 가지게 된다.

![](https://velog.velcdn.com/images/seonydg/post/f07e9873-a01b-4060-8c99-e38c79e7408c/image.png)


## SPVNAS

SPVNAS에서 가장 중요하다고 볼 수 있는 것은 Sparse Point Convoution이라는 새로운 Operation이다.
그리고 여기에서 Efficiency를 조금 증가시키기 위해 NAS를 제안한다. 

아래의 그림에서 알 수 있듯이, 형상이 드러나지 않는다는 단점을 가지고 있다. 그래서 classes를 구분하는데 있어서 어려움을 지닌다. 그렇기 때문에 Point Voxel Convolution이라는 제안이 기존에 있었다.

![](https://velog.velcdn.com/images/seonydg/post/1d59bc9b-2e28-40b4-aac9-3f472f7ee1d5/image.png)

Point Voxel Convolution(PVConv, NeurlPS2019)은 복셀을 사용하기에 효율적이긴 하지만 Dense voxel grid를 사용한다. Dense voxel grid의 경우에는 모든 공간에 데이터가 있다는 전자하에 연산을 하기에 매우 느리다. 이에 반해서 Sparse는 데이터가 있는 공간이 굉장히 적기 때문에 연산이 빠르다.
그럼에도 불구하고 Voxelization을 통해서 연산을 하기에 하나하나의 point의 이웃 데이터를 찾아 연산을 하는 것보다는 빨랐기 때문에 Inference Speed가 빠르다.
순서는 아래와 같은데, MLP과정 이후 Neighbor Search가 없기에 매우 느린 연산 구간은 없다. 그리고 마지막에 voxel feature와 point feature를 합쳐서 최종 Output을 내뱉는다.

![](https://velog.velcdn.com/images/seonydg/post/5ce496df-ce2e-4f51-a450-fc4e41b04add/image.png)

SPVConv는 PVConv에서 Sparse만 추가가 된 버전이다. 
그래서 SPVNAS는 Sparse Convolution을 잘 이해하고 적용하는 것이 중요하다고 볼 수 있다.

![](https://velog.velcdn.com/images/seonydg/post/4380d0c7-1469-4dd6-8f0c-c1f7c0b5ed25/image.png)


## NAS(Network Architecture Search)

NAS를 간단히 보면, 먼저 사람이 정해놓은 Search Space가 있고 Search Space를 탐색할 Search Strategy를 정한다. 그 후에 Search Strategy 통해서 찾은 아키텍쳐를 찾고 performance를 측정한다. 그리고 performance를 리워드로 사용하거나 이 performance를 기준으로 다른 Search Strategy에서 새로운 아키텍쳐를 찾아내는 방법을 반복한다.

![](https://velog.velcdn.com/images/seonydg/post/c794b872-dacc-46fc-8f9d-748f6a76a82e/image.png)

매우 많은 하이퍼 파라미터들을 조정하면서 가장 성능이 좋거나 문제 해결에 맞는 원하는 모델을 찾는 방법이다.

![](https://velog.velcdn.com/images/seonydg/post/a96df109-97b5-4966-8342-0ea4c093ea4d/image.png)



## Results

![](https://velog.velcdn.com/images/seonydg/post/ba10b5b6-6876-408a-8107-993366d40abf/image.png)
