"""leave-one-person-out으로 평가하는 CNN pipeline (PyTorch). baseline_linear.py와 비교용이다.

입력: 눈 이미지 (1, 36, 60) + 2DHead(yaw, pitch) -> 출력: 2DGaze(yaw, pitch). 손실은 라디안
단위 MSE, 평가 지표는 baseline_linear.py와 같은 각도 오차(도)다. fold마다 모델을 새로 초기화하고,
같은 하이퍼파라미터를 모든 fold에 쓴다 (README의 leave-one-person-out 규칙).

    python train_cnn.py                       # 15 fold 전부, 기본 하이퍼파라미터
    python train_cnn.py --folds p00 p01        # 일부 fold만 (빠른 점검용)
    python train_cnn.py --epochs 3 --folds p00 # 스모크 테스트
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

import reader
from baseline_linear import PERSONS, angular_error
from model import GazeNet


def run_epoch(model, loader, device, optimizer=None):
  """optimizer가 있으면 학습, 없으면 평가. (평균 loss, 예측 전부, 정답 전부)를 돌려준다."""
  train = optimizer is not None
  model.train(train)
  criterion = nn.MSELoss()

  total_loss, n = 0.0, 0
  preds, gts = [], []
  for info, label in loader:
    eye = torch.from_numpy(info["eye"]).to(device)
    head = torch.from_numpy(info["head_pose"]).to(device)
    gaze = torch.from_numpy(label).to(device)

    with torch.set_grad_enabled(train):
      pred = model(eye, head)
      loss = criterion(pred, gaze)

    if train:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    bs = eye.shape[0]
    total_loss += loss.item() * bs
    n += bs
    preds.append(pred.detach().cpu().numpy())
    gts.append(gaze.detach().cpu().numpy())

  return total_loss / n, np.concatenate(preds), np.concatenate(gts)


def run_fold(test_person, root, epochs, batch_size, lr, num_workers, device, seed):
  torch.manual_seed(seed)
  train_persons = [p for p in PERSONS if p != test_person]
  label = lambda p: os.path.join(root, "Label", f"{p}.label")
  image_root = os.path.join(root, "Image")

  train_loader = reader.txtload([label(p) for p in train_persons], image_root,
                                 batch_size=batch_size, shuffle=True, num_workers=num_workers)
  test_loader = reader.txtload(label(test_person), image_root,
                                batch_size=batch_size, shuffle=False, num_workers=num_workers)

  model = GazeNet().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  for epoch in range(epochs):
    t0 = time.time()
    train_loss, _, _ = run_epoch(model, train_loader, device, optimizer)
    print(f"    epoch {epoch + 1}/{epochs}  train MSE(rad^2)={train_loss:.5f}  ({time.time() - t0:.0f}s)")

  _, preds, gts = run_epoch(model, test_loader, device)
  err = angular_error(preds, gts).mean()
  return err


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--root", default="data/EyeBased/MPIIGaze")
  p.add_argument("--folds", nargs="*", default=None,
                  help="평가할 test person만 지정 (기본: 15명 전부)")
  p.add_argument("--epochs", type=int, default=8)
  p.add_argument("--batch-size", type=int, default=256)
  p.add_argument("--lr", type=float, default=1e-3)
  p.add_argument("--num-workers", type=int, default=4)
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  folds = args.folds if args.folds else PERSONS
  print(f"device={device}  folds={folds}  epochs={args.epochs}  batch_size={args.batch_size}")

  errors = {}
  for test in folds:
    print(f"== fold: test on {test} ==")
    t0 = time.time()
    err = run_fold(test, args.root, args.epochs, args.batch_size, args.lr,
                    args.num_workers, device, args.seed)
    errors[test] = err
    print(f"  test {test}: {err:.2f}  ({time.time() - t0:.0f}s total)")

  print()
  print("| person | " + " | ".join(errors.keys()) + " | mean |")
  print("|---" * (len(errors) + 2) + "|")
  print("| angular error (deg) | " + " | ".join(f"{v:.2f}" for v in errors.values())
        + f" | **{np.mean(list(errors.values())):.2f}** |")


if __name__ == "__main__":
  main()
