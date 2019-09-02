
(현재 진행중)

# BigBiGAN

간단히 표현하면 BiGAN에 당시 이미지 생성에서 SOTA였던 BigGAN의 아키텍쳐를 사용한 모델 (나중엔 바뀔테니 과거형으로)

기본적인 구조는 위에 나와있듯이 BigGAN의 아키텍쳐를 따르고 (BigGAN은 또 SAGAN의 아키텍쳐를 따른다고한다. 덕분에 논문 3개보는중)

입력부분이 x 에서 z 를 예측하는 Encoder와 z에서 x를 생성하는 Generator로 이루어진다 (CycleGAN과 비슷한 느낌?)


Discriminator 부분도 3가지 파트로 나뉘어 있는데 

x만을 입력받아 $S_x$를 출력하는 F (ConvNet)
z만을 입력받아 $S_z$를 출력하는 H (MLP)
그 두가지의 출력으로 Sxz를 출력하는 J (MLP) 

로 이루어지고 

손실은 Generator에선 3가지의 합을
Discriminator에선 3가지의 힌지의 합을 사용한다
Y 값은 0~1 이 아닌 -1 ~ 1로 사용

그외엔 모델링에서의 특별한 점은 없는듯 하다
CReLU 라는 렐루와 -렐루의 Concat 을 사용한 참신한 기법을 썻다는거 빼고는
shape가 변화하긴하나 그래도 정보손실이 없는 Relu라는 점에서 참신하다
2016년에 등재된 논문에 나온거였는데 처음보다니.. 공부가 부족하다...




이부분은 분리될 가능성이 높음
# BigGAN



# SAGAN 
Self Attention GAN으로 정말 Transformer 이후 어텐션은 안쓰이는곳이 없는거 같다...
하긴 직관적으로 생각해도 이미지같은 중요한부분과 안중요한 부분이 확 나눠지는 데이터에서
어텐션이 잘 작동하는건 어떻게보면 당연한게 아닐까 한다.

특별히 사용된 기법은 
Spectral Normalize 라는 기법?
학습의 안정화를 위해 사용한다고 한다.
완전히 이해된건 아닌데 행렬의 고유값(sigular value)의 최대치를 제한하는 Lipschitz 상수를 사용한다고 한다.
저 상수는 기울기의 최대값을 제한하는걸로 아는데 여기서도 사용되나 보다.
비정상적으로 큰 기울기를 생성되지 않게 하는 의도인것 같은데
행렬의 고유값에 응용에 대해선 조금더 공부가 필요하겠다.

TTUR이라는 Generator와 Discriminator의 학습율을 다르게 하는 기법을 사용해
Generator 가 여러번 학습할때 Discriminator가 한번씩 학습할 수 있도록 변형하여
정규화된 Discriminator가 

평가측도는 기존모델과의 비교를 위해 IS를 
그리고 IS에서 평가하지 못하는 부분의 평가를위해 FID를 사용

## 모델 구조






난 왜 자꾸 학습량이 엄청난 모델들만 고르는걸까....






## 해야할일
1. EG 구현
 - Self Attention 구현
2. D 구현
3. loss 구현
4. **Encorder에 사용될 ResNet50구현** 
 - 왜했는진 모르겠지만 갑자기 해보고 싶었다. 덕분에 ResNet도 50layer 이상에선 Bottle Neck 구조를 사용한다는것을 배움..
