# 범죄 탐지 소형 로봇 (TransBot-Anomaly-Tracker)
주택단지 실시간 모니터링을 위한 범죄 탐지 소형 로봇 프로젝트입니다. 본 프로젝트는 AI와 로봇 기술을 활용하여 
실시간 범죄 행동을 탐지하고, 이를 바탕으로 안전한 주거 환경을 제공합니다.

## 프로젝트 배경 및 목적
- **배경**:
  경찰 미래비전 2050에서는 AI와 로봇 기술을 활용한 첨단 치안 활동을 목표로 하고 있습니다. 그러나 기존의 드론 및 4족 보행 로봇은 
  높은 비용, 개인정보 침해, 안전 문제와 같은 한계가 존재합니다.
- **목적**:
  소형 4륜 구동 로봇을 사용하여 보다 효율적이고 안전한 순찰 지원 및 범죄 예방 활동을 목표로 합니다. 이를 통해 CCTV 사각지대와 
  접근이 어려운 구역에서의 범죄 탐지를 지원하고, 경찰의 업무 부담을 줄이는 데 기여합니다.


## 주요 기능
1. **모바일 로봇의 지정 경로 순찰**:
   ROS와 라이다 기술을 활용하여 자율적으로 지정된 구역을 순찰.
2. **범죄 행동 탐지 및 대응**:
   OpenCV, YOLO, CNN 등을 사용하여 실시간으로 이상 행동을 탐지하고 경고 대응.
3. **범죄 상황 요약**:
   LLM(대규모 언어 모델)을 활용해 상황을 요약하고 보고서 작성.


#### 5. **기술 스택**
## 기술 스택
- **Python**: 데이터 처리 및 AI 모델 구현.
- **TensorFlow & PyTorch**: 딥러닝 기반 모델 학습 및 추론.
- **ROS (Robot Operating System)**: 로봇 자율 주행 및 경로 탐색.
- **OpenCV & YOLO**: 이미지 처리 및 객체 탐지.
- **LLM & RAG**: 생성형 AI 기반 요약 및 상황 기록.

---

# AI 모델
## 프로젝트 배경 및 모델 선정 과정
1. **객체 탐지**: 실시간으로 보행자, 차량, 기타 객체를 탐지합니다.
2. **행동 분석**: 시간 기반 행동 패턴을 분석하여 이상 행동을 탐지합니다.
3. **효율성**: Jetson Nano와 같은 경량 하드웨어 환경에서도 실행 가능한 시스템을 구축합니다.

## 모델 선정 과정

### 1. **초기 프로젝트 목표**
시스템 설계 초기에는 다음과 같은 목표를 설정했습니다:
- 실시간 객체 탐지와 분류 수행.
- 시간 기반 행동 패턴을 분석해 이상 행동을 탐지.
- 경량 디바이스에서도 효율적으로 동작.

### 2. **모델 선정 기준**
모델을 선정할 때 다음 조건을 고려했습니다:
- **실시간성**: 시간 지연 없이 즉각적으로 처리 가능.
- **정확성**: 높은 탐지 및 분류 정확도.
- **경량성**: Jetson Nano와 같은 제한된 자원 환경에서 실행 가능.
- **시간 분석**: 시간 기반 행동을 효과적으로 분석 가능.

### 3. **초기 검토된 모델**
| **모델 조합**         | **장점**                                       | **단점**                                          |
|-----------------------|-----------------------------------------------|--------------------------------------------------|
| EfficientNet + GRU    | 높은 특징 추출 정확도                          | 높은 계산 비용, 실시간 처리 속도 부족              |
| MobileNet + LSTM      | 경량화된 특징 추출                             | 직접적인 객체 탐지 불가, 추가 알고리즘 필요         |
| 3D CNN + GRU          | 시간-공간 분석 결합                            | 높은 계산량, Jetson Nano 환경에서 비효율적          |

### 4. **최종 모델 선정**
위 옵션들의 한계를 검토한 후, 다음 모델을 최종 선정했습니다:
- **YOLO (Tiny 버전)**: 실시간 객체 탐지와 특징 추출을 동시에 수행.
- **GRU/LSTM/TSM**: 시간적 패턴 분석 및 행동 분류.

## 최종 모델 구성

### 1. **YOLO**
- **역할**: 각 프레임에서 객체 탐지 및 특징 추출.
- **선정 이유**:
  - 객체 탐지와 특징 추출을 결합한 효율적인 구조.
  - 경량화된 버전(YOLOv4-Tiny, YOLOv8-Nano)으로 Jetson Nano에서도 실시간 동작 가능.

### 2. **GRU**
- **역할**: 단기 시간적 패턴 분석.
- **선정 이유**:
  - LSTM에 비해 경량화되어 있음.
  - 짧은 시간 간격의 행동 변화를 효과적으로 학습.

### 3. **LSTM**
- **역할**: 장기 시간 의존성 분석.
- **선정 이유**:
  - 복잡한 반복 행동 패턴 분석에 적합.

### 4. **TSM**
- **역할**: 경량화된 시간 분석.
- **선정 이유**:
  - 추가 파라미터 없이 실시간 시스템에서 시간적 정보를 효율적으로 학습 가능.

---

## 워크플로우

```mermaid
graph TD
A[입력 영상] --> B[YOLO: 객체 탐지]
B --> C[객체 위치 및 클래스 정보 추출]
C --> D[시간 분석: GRU/LSTM/TSM]
D --> E[행동 분류]
