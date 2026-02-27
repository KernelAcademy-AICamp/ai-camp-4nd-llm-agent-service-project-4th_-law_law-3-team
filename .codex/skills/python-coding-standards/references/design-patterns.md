# Python 디자인 패턴 가이드

상황별 적합한 디자인 패턴과 Python 구현 예제.

## 목차

1. [생성 패턴](#생성-패턴)
2. [구조 패턴](#구조-패턴)
3. [행동 패턴](#행동-패턴)
4. [선택 가이드](#선택-가이드)

---

## 생성 패턴

### Builder 패턴

**사용 시점**: 복잡한 객체를 단계별로 생성해야 할 때

```python
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Email:
    recipient: str
    subject: str
    body: str
    cc: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)

class EmailBuilder:
    def __init__(self):
        self._recipient: Optional[str] = None
        self._subject: str = ""
        self._body: str = ""
        self._cc: List[str] = []
        self._attachments: List[str] = []

    def to(self, recipient: str) -> "EmailBuilder":
        self._recipient = recipient
        return self

    def with_subject(self, subject: str) -> "EmailBuilder":
        self._subject = subject
        return self

    def with_body(self, body: str) -> "EmailBuilder":
        self._body = body
        return self

    def add_cc(self, cc: str) -> "EmailBuilder":
        self._cc.append(cc)
        return self

    def attach(self, file_path: str) -> "EmailBuilder":
        self._attachments.append(file_path)
        return self

    def build(self) -> Email:
        if not self._recipient:
            raise ValueError("Recipient is required")
        return Email(
            recipient=self._recipient,
            subject=self._subject,
            body=self._body,
            cc=self._cc,
            attachments=self._attachments,
        )

# 사용 예
email = (
    EmailBuilder()
    .to("user@example.com")
    .with_subject("회의 안내")
    .with_body("내일 오후 2시 회의가 있습니다.")
    .add_cc("manager@example.com")
    .build()
)
```

### Factory 패턴

**사용 시점**: 조건에 따라 다른 타입의 객체를 생성해야 할 때

```python
from abc import ABC, abstractmethod
from typing import Dict, Type

class Notification(ABC):
    @abstractmethod
    def send(self, message: str) -> None:
        pass

class EmailNotification(Notification):
    def send(self, message: str) -> None:
        print(f"이메일 발송: {message}")

class SMSNotification(Notification):
    def send(self, message: str) -> None:
        print(f"SMS 발송: {message}")

class PushNotification(Notification):
    def send(self, message: str) -> None:
        print(f"푸시 알림: {message}")

class NotificationFactory:
    _notifications: Dict[str, Type[Notification]] = {
        "email": EmailNotification,
        "sms": SMSNotification,
        "push": PushNotification,
    }

    @classmethod
    def create(cls, notification_type: str) -> Notification:
        notification_class = cls._notifications.get(notification_type)
        if not notification_class:
            raise ValueError(f"Unknown notification type: {notification_type}")
        return notification_class()

# 사용 예
notification = NotificationFactory.create("email")
notification.send("안녕하세요!")
```

### Singleton 패턴

**사용 시점**: 인스턴스가 하나만 필요할 때 (설정, 로거, DB 연결 풀 등)

```python
# 방법 1: 모듈 레벨 (권장 - 가장 Pythonic)
# config.py
class _Config:
    def __init__(self):
        self.debug = False
        self.database_url = ""

config = _Config()  # 모듈 import 시 한 번만 생성

# 방법 2: __new__ 사용
class DatabaseConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.connection = None  # 실제 연결 로직
```

---

## 구조 패턴

### Decorator 패턴

**사용 시점**: 기존 함수/클래스에 기능을 추가할 때

```python
import functools
import time
from typing import Callable, Any

# 함수 데코레이터 - 실행 시간 측정
def timing_decorator(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 실행 시간: {end_time - start_time:.4f}초")
        return result
    return wrapper

# 함수 데코레이터 - 재시도 로직
def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

# 사용 예
@timing_decorator
@retry(max_attempts=3)
def fetch_data(url: str) -> dict:
    # API 호출 로직
    pass
```

### Facade 패턴

**사용 시점**: 복잡한 서브시스템을 단순한 인터페이스로 제공할 때

```python
class VideoConverter:
    def convert(self, filename: str, format: str) -> str:
        return f"{filename}.{format}"

class AudioExtractor:
    def extract(self, video_file: str) -> str:
        return f"{video_file}_audio.mp3"

class Uploader:
    def upload(self, file: str, platform: str) -> str:
        return f"https://{platform}.com/{file}"

# Facade - 복잡한 작업을 단순화
class MediaPublisher:
    def __init__(self):
        self._converter = VideoConverter()
        self._extractor = AudioExtractor()
        self._uploader = Uploader()

    def publish_video(self, filename: str, platform: str) -> str:
        """비디오 변환, 오디오 추출, 업로드를 한 번에 처리."""
        converted = self._converter.convert(filename, "mp4")
        self._extractor.extract(converted)  # 팟캐스트용
        return self._uploader.upload(converted, platform)

# 사용 예 - 복잡한 내부 로직을 알 필요 없음
publisher = MediaPublisher()
url = publisher.publish_video("my_video", "youtube")
```

### Adapter 패턴

**사용 시점**: 호환되지 않는 인터페이스를 연결할 때

```python
from abc import ABC, abstractmethod

# 기존 레거시 시스템
class LegacyPaymentSystem:
    def process_payment_legacy(self, amount: int, currency: str) -> bool:
        print(f"레거시 결제: {amount} {currency}")
        return True

# 새로운 인터페이스
class PaymentProcessor(ABC):
    @abstractmethod
    def pay(self, amount: float) -> bool:
        pass

# 어댑터
class LegacyPaymentAdapter(PaymentProcessor):
    def __init__(self, legacy_system: LegacyPaymentSystem):
        self._legacy = legacy_system

    def pay(self, amount: float) -> bool:
        # 새 인터페이스를 레거시 시스템에 맞게 변환
        amount_in_cents = int(amount * 100)
        return self._legacy.process_payment_legacy(amount_in_cents, "KRW")

# 사용 예
legacy = LegacyPaymentSystem()
processor = LegacyPaymentAdapter(legacy)
processor.pay(10000.0)  # 새 인터페이스로 호출
```

---

## 행동 패턴

### Strategy 패턴

**사용 시점**: 알고리즘을 런타임에 교체해야 할 때

```python
from typing import Callable, List

# Python에서는 함수를 일급 객체로 전달 (간단한 경우)
def bubble_sort(data: List[int]) -> List[int]:
    # 버블 정렬 구현
    return sorted(data)

def quick_sort(data: List[int]) -> List[int]:
    # 퀵 정렬 구현
    return sorted(data)

class DataProcessor:
    def __init__(self, sort_strategy: Callable[[List[int]], List[int]]):
        self._sort = sort_strategy

    def process(self, data: List[int]) -> List[int]:
        return self._sort(data)

# 사용 예 - 전략 교체
processor = DataProcessor(bubble_sort)
result = processor.process([3, 1, 4, 1, 5])

processor = DataProcessor(quick_sort)  # 전략 변경
result = processor.process([3, 1, 4, 1, 5])
```

### State 패턴

**사용 시점**: 객체의 상태에 따라 동작이 달라질 때

```python
from abc import ABC, abstractmethod

class OrderState(ABC):
    @abstractmethod
    def next(self, order: "Order") -> None:
        pass

    @abstractmethod
    def cancel(self, order: "Order") -> None:
        pass

class PendingState(OrderState):
    def next(self, order: "Order") -> None:
        print("주문 확정됨")
        order.state = ConfirmedState()

    def cancel(self, order: "Order") -> None:
        print("주문 취소됨")
        order.state = CancelledState()

class ConfirmedState(OrderState):
    def next(self, order: "Order") -> None:
        print("배송 시작")
        order.state = ShippedState()

    def cancel(self, order: "Order") -> None:
        print("확정된 주문은 취소 불가")

class ShippedState(OrderState):
    def next(self, order: "Order") -> None:
        print("배송 완료")
        order.state = DeliveredState()

    def cancel(self, order: "Order") -> None:
        print("배송 중 취소 불가")

class DeliveredState(OrderState):
    def next(self, order: "Order") -> None:
        print("이미 배송 완료")

    def cancel(self, order: "Order") -> None:
        print("반품 프로세스 시작")

class CancelledState(OrderState):
    def next(self, order: "Order") -> None:
        print("취소된 주문")

    def cancel(self, order: "Order") -> None:
        print("이미 취소됨")

class Order:
    def __init__(self):
        self.state: OrderState = PendingState()

    def next(self) -> None:
        self.state.next(self)

    def cancel(self) -> None:
        self.state.cancel(self)

# 사용 예
order = Order()
order.next()    # 주문 확정됨
order.next()    # 배송 시작
order.cancel()  # 배송 중 취소 불가
```

### Observer 패턴

**사용 시점**: 이벤트 기반으로 여러 객체에 알림을 보낼 때

```python
from abc import ABC, abstractmethod
from typing import List

class Observer(ABC):
    @abstractmethod
    def update(self, event: str, data: dict) -> None:
        pass

class Subject:
    def __init__(self):
        self._observers: List[Observer] = []

    def attach(self, observer: Observer) -> None:
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def notify(self, event: str, data: dict) -> None:
        for observer in self._observers:
            observer.update(event, data)

# 구체적인 옵저버들
class EmailNotifier(Observer):
    def update(self, event: str, data: dict) -> None:
        print(f"이메일 발송: {event} - {data}")

class SlackNotifier(Observer):
    def update(self, event: str, data: dict) -> None:
        print(f"슬랙 알림: {event} - {data}")

class LogRecorder(Observer):
    def update(self, event: str, data: dict) -> None:
        print(f"로그 기록: {event} - {data}")

# 사용 예
order_system = Subject()
order_system.attach(EmailNotifier())
order_system.attach(SlackNotifier())
order_system.attach(LogRecorder())

order_system.notify("ORDER_CREATED", {"order_id": 123, "total": 50000})
```

### Iterator 패턴 (Generator)

**사용 시점**: 컬렉션을 순회할 때, 특히 대용량 데이터

```python
from typing import Generator, Iterator

# Generator 함수 사용 (권장)
def read_large_file(file_path: str) -> Generator[str, None, None]:
    """대용량 파일을 한 줄씩 읽기 - 메모리 효율적."""
    with open(file_path, "r") as f:
        for line in f:
            yield line.strip()

# Generator 표현식
def get_even_numbers(limit: int) -> Generator[int, None, None]:
    return (x for x in range(limit) if x % 2 == 0)

# 클래스 기반 Iterator
class DateRange:
    def __init__(self, start_date, end_date):
        self.start = start_date
        self.end = end_date
        self.current = start_date

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        result = self.current
        self.current += timedelta(days=1)
        return result

# 사용 예
for line in read_large_file("huge_file.txt"):
    process(line)  # 한 번에 한 줄만 메모리에 로드
```

---

## 선택 가이드

| 문제 상황 | 추천 패턴 |
|----------|----------|
| 객체 생성이 복잡하고 단계가 많다 | Builder |
| 조건에 따라 다른 타입의 객체 생성 | Factory |
| 전역적으로 하나의 인스턴스만 필요 | Singleton (모듈 레벨) |
| 기존 함수에 로깅/캐싱/검증 추가 | Decorator |
| 복잡한 시스템을 단순하게 사용 | Facade |
| 서로 다른 인터페이스 연결 | Adapter |
| 알고리즘을 런타임에 교체 | Strategy |
| 객체 상태에 따라 동작 변경 | State |
| 이벤트 발생 시 여러 객체에 알림 | Observer |
| 대용량 데이터 순회 | Iterator (Generator) |

### 주의사항

- **YAGNI**: 필요할 때만 패턴 적용
- **단순함 우선**: Python 내장 기능으로 충분하면 패턴 불필요
- **일관성**: 프로젝트에서 사용 중인 패턴 따르기
- **문서화**: 패턴 사용 이유를 주석으로 설명
