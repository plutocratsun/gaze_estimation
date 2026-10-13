"""눈 이미지 + head pose로 2DGaze를 회귀하는 작은 CNN.

baseline_linear.py의 ridge 회귀(평균 7.27도)와 비교하기 위한 비선형 pipeline이다.

구조: conv-relu-pool 블록 2개로 눈 이미지(1, 36, 60)에서 특징을 뽑고, 2DHead(yaw, pitch)를
이어 붙인 뒤 fully-connected 층 2개로 2DGaze(yaw, pitch)를 바로 예측한다 (분류가 아닌 회귀이므로
마지막 층에 활성화 함수를 두지 않는다).
"""
import torch
import torch.nn as nn


class GazeNet(nn.Module):
  def __init__(self, head_dim=2):
    super().__init__()
    # 입력 (N, 1, 36, 60)
    self.conv = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
      nn.MaxPool2d(2),                                   # -> (16, 18, 30)
      nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
      nn.MaxPool2d(2),                                   # -> (32, 9, 15)
      nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
      nn.AdaptiveAvgPool2d((4, 5)),                      # -> (64, 4, 5), 입력 크기 변화에 안전
    )
    feat_dim = 64 * 4 * 5
    self.fc = nn.Sequential(
      nn.Linear(feat_dim + head_dim, 128), nn.ReLU(inplace=True),
      nn.Dropout(0.3),
      nn.Linear(128, 2),
    )

  def forward(self, eye, head_pose):
    x = self.conv(eye)
    x = x.flatten(1)
    x = torch.cat([x, head_pose], dim=1)
    return self.fc(x)
