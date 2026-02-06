"""
GPU 온도 모니터링

nvidia-smi 기반 GPU 온도 실시간 모니터링 및 자동 쿨다운 대응.

하드웨어 프로필별 동작:
- Desktop (5060Ti): 모니터링 불필요 (충분한 쿨링)
- Laptop (3060): 85°C 임계값, 자동 배치 축소 + 쿨다운
- Mac (M3): 온도 모니터링 불필요 (MPS 백엔드)
"""

import subprocess
import time
from dataclasses import dataclass, field


@dataclass
class ThermalState:
    """GPU 열 상태"""

    temperature: float = 0.0
    threshold: int = 85
    consecutive_over: int = 0
    max_consecutive: int = 3
    batch_reduced: bool = False
    original_batch_size: int = 0

    @property
    def is_critical(self) -> bool:
        """연속 임계값 초과 시 위험 상태"""
        return self.consecutive_over >= self.max_consecutive

    @property
    def is_over_threshold(self) -> bool:
        return self.temperature >= self.threshold

    @property
    def is_recovered(self) -> bool:
        """70°C 미만으로 회복"""
        return self.temperature < 70.0


@dataclass
class TemperatureMonitor:
    """
    GPU 온도 모니터 (nvidia-smi 기반)

    자동 대응 로직:
    1. threshold 도달 → 배치 크기 50% 감소 + 30초 대기
    2. 연속 3회 초과 → 자동 중지 (체크포인트 저장 권장)
    3. 70°C 미만 회복 → 배치 크기 복원
    """

    threshold: int = 85
    cooldown_seconds: int = 30
    _state: ThermalState = field(init=False)

    def __post_init__(self) -> None:
        self._state = ThermalState(threshold=self.threshold)

    def get_gpu_temperature(self) -> float:
        """nvidia-smi로 GPU 온도 조회 (섭씨)"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split("\n")[0])
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return 0.0

    def check_and_adjust(self, current_batch_size: int) -> tuple[int, bool]:
        """
        온도 체크 후 배치 크기 조정

        Args:
            current_batch_size: 현재 배치 크기

        Returns:
            (조정된 배치 크기, 중지 필요 여부)
        """
        temp = self.get_gpu_temperature()
        self._state.temperature = temp

        if temp <= 0.0:
            # nvidia-smi 미설치 또는 조회 실패
            return current_batch_size, False

        if self._state.is_over_threshold:
            self._state.consecutive_over += 1
            print(
                f"[THERMAL] GPU 온도: {temp:.0f}°C >= {self.threshold}°C "
                f"(연속 {self._state.consecutive_over}회)"
            )

            if self._state.is_critical:
                print("[THERMAL] 연속 임계값 초과! 자동 중지를 권장합니다.")
                return current_batch_size, True

            # 배치 크기 50% 감소
            if not self._state.batch_reduced:
                self._state.original_batch_size = current_batch_size
                self._state.batch_reduced = True

            reduced = max(current_batch_size // 2, 1)
            print(
                f"[THERMAL] 배치 크기 축소: {current_batch_size} → {reduced}, "
                f"{self.cooldown_seconds}초 대기"
            )
            time.sleep(self.cooldown_seconds)
            return reduced, False

        elif self._state.batch_reduced and self._state.is_recovered:
            # 온도 회복 → 배치 크기 복원
            restored = self._state.original_batch_size
            self._state.batch_reduced = False
            self._state.consecutive_over = 0
            print(
                f"[THERMAL] GPU 온도 회복: {temp:.0f}°C, "
                f"배치 크기 복원: {current_batch_size} → {restored}"
            )
            return restored, False

        else:
            self._state.consecutive_over = 0
            return current_batch_size, False

    @property
    def state(self) -> ThermalState:
        return self._state
