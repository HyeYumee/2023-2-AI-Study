#<week1>
DL Component
1. Data : 목표 Task에 의존함.
    > type : Classification / Semantic Segmentation / Object Detection / Pose Estimation 
2. Model : 입력 > feature 추출 > 출력 변환
3. Loss function : 학습 중 알고리즘이 얼마나 잘못 예측하는지에 대한 지표, 알고리즘이 예측한 값과 실제 정답 차이를 비교해 학습
    > Regression Task(평균제곱오차) / Classification Task / Probabilicastic Task(MLE)
4. Optimization / Regularization
    > Optimization에는 Gradient Descent Method(경사하강법)을 통해 loss function을 빠르고 정확하게 줄임(보통 Adam, AdamW 사용)
    > Regularization은 학습을 방해하여 일반화 성능을 높임.


Neural Network
> Function Approximaotrs
> 활성함수(Activation Function, layer 내에서 node의 값을 변형, 주로 Sigmoid함수 사용해서 값을 0~1 사이로 압축해주고, 모델을 곡선형태로 만들어 정교한 예측 가능)로 비선형 함수를 사용하는데, 이는 활성함수가 선형 함수라면 여러겹이 쌓여도 하나의 함수로 대체가 가능하기 때문이다.


Multi-Layer Perceptron(MLP)
> Loss function이 0이 되었더라도 noise가 많은 학습데이터를 간과할 수 없으므로 항상 target을 위해 loss function이 제 기능을 한다고 할 수 없음.


Generalization
> 대부분 일반화 성능을 높이는 것을 목표로 함. Traning error가 0이 되더라도 최적의 값에 도달하는 것은 아님. Test error가 Training error를 따라 감소하다가 어느 순간부터 증가추세로 바뀌기 때문.
1. Overfitting(학습 데이터상에서는 잘 동작하지만 테스트 데이터상으로는 잘 동작하지 않음)
2. Underfitting(네트워크가 너무 단순하거나 학습 데이터에도 부합하지 못하는 경우)

#Cross Validation
> 일반적으로 데이터는 Train Data(Training data와 Valid data로 나눔)와 Test Data로 구분되어 제공, 학습이 잘 되고 있는지를 확인하는 지표를 활용하여 검증

#Ensemble
> 여러개의 분류모델을 조합해 더 나은 성능의 모델을 도출.
1. Bagging : subset을 나누어 학습, 각각의 voting이나 averaging을 구함(병렬학습)
2. Boosting : 학습이 제대로 이루어지지 않은 데이터들을 모아서 새로운 간단한 모델로 재학습(순차학습)

#Regularization
Early Stopping / Parameter norm penalty / Data augmentation(한정 데이터를 활용해 변형하여 데이터의 양을 늘림(각도 조정, crop 등), 단 이는 데이터의 특성에 의존적임.9와 6을 뒤집으면 같은 숫자가 된다는 점이 예.이를 보완하기 위해 labeling을 함께 수정.)

#Noise robustness : 이상치, 노이즈가 들어와도 안정적인 모델(가우시안, 라플라스)

#DropOut : Train 과정에서만 적용, 일정 확률로 특정 node를 학습에 참여하지 않도록 하는 방법, Test에서는 드랍되는 것 없이 모든 node가 항상 추론에 참여해야함.

#Label Smoothing : 모델이 너무 확신을 가지지 않도록 도와줌.
> 일반적으로 모델은 예측시 클래스에 대해 확률을 1과 0 이분법적으로 예측.(과적합 초래)
> [0,1,0,0] > [0.025,0.925,0.025,0.025] 일반화 성능 향상


Convolution Neural Network(CNN, 합성곱신경망)
> 이미지의 공간 정보를 유지하며 학습
> 이전에는 FNN 사용, 이는 인접 픽셀간 상관관계가 무시되어 이미지를 벡터화하는 과정에서 정보손실발생하는 문제가 있었음. 이를 CNN이 해결
> Convolutional Layer / Pooling 반복, 5x5 image를 3x3 filter를 활용해서 stemping, 3x3 영역을 1x1 값으로 변형연산, 결국 3x3 convolved feature 생성됨 / filter는 output의 depth만큼 개수가 들어가야 함.
> Pooling은 size를 줄임, Max Pooling(가장 큰 값), Average Pooling(평균값) 
ex. 1x1 convolution : 1x1 filter 사용. depth는 origin data의 depth와 같아야함.


Modern CNN
#AlexNet
> 2개의 네트워크, 11x11 filter, convolution layer x 5, dense layer x 3
> ReLU(activation function 종류)

#VGGNet
> 3x3 filter, 같은 receptive filed에서 계산량 줄이기

#GoogLeNet
> 1x1 filter

#ResNet
> 사람의 능력을 뛰어넘은 첫번째 모델
> Neural Network가 깊어질 수록 학습하기가 어려웠음. 역전파 과정에서 활성화함수의 미분값을 곱하다보니(기울기가 존재하지 않는 경우에 직면) 가중치 갱신이 멈춤. 이를 해결하기 위해 skip connection을 도입, 아무리 미분해도 기울기 1이 남기에 기울기 소실 문제를 해결.