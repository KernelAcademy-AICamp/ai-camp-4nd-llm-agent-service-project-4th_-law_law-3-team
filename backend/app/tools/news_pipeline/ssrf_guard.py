"""SSRF 방어: httpx 요청 전 내부망 IP 차단"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# 차단 대상 IP 대역 (내부망, 로컬호스트, 링크-로컬)
_BLOCKED_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv6Network("::1/128"),
    ipaddress.IPv6Network("fc00::/7"),
    ipaddress.IPv6Network("fe80::/10"),
]


def validate_url(url: str) -> bool:
    """URL이 외부 접근 가능한지 검증. 내부망 IP면 False 반환."""
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return False

    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in addr_info:
            ip = ipaddress.ip_address(sockaddr[0])
            for network in _BLOCKED_NETWORKS:
                if ip in network:
                    logger.warning("SSRF 차단: %s → %s (내부망)", url, ip)
                    return False
    except (socket.gaierror, ValueError):
        logger.warning("DNS 확인 실패: %s", hostname)
        return False

    return True
