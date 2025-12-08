# rate_limiter.py
import json
import time
from pathlib import Path
from typing import Set, Dict, List, Tuple
from collections import defaultdict
from datetime import datetime
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from send_mail import SPCMailer

class RateLimiter:
    def __init__(
        self,
        blocklist_file: str = "blocklist.json",
        rate_limit: int = 5,
        time_window: int = 60
    ):
        """
        Rate limiter with permanent blocking for abusive users.
        
        Args:
            blocklist_file: Path to JSON file storing blocked IPs
            rate_limit: Maximum allowed requests in time_window
            time_window: Time window in seconds
        """
        self.blocklist_file = Path(blocklist_file)
        self.rate_limit = rate_limit
        self.time_window = time_window
        
        # In-memory tracking
        self.request_logs: Dict[str, List[float]] = defaultdict(list)
        self.blocked_ips: Set[str] = set()
        
        # Load existing blocklist
        self.load_blocklist()
        
        logging.info(f"RateLimiter initialized: {rate_limit} requests per {time_window} seconds")
    
    def load_blocklist(self) -> None:
        """Load blocked IPs from JSON file, create if doesn't exist"""
        try:
            if self.blocklist_file.exists():
                with open(self.blocklist_file, 'r') as f:
                    data = json.load(f)
                    self.blocked_ips = set(data.get("blocked_ips", []))
                    logging.info(f"Loaded {len(self.blocked_ips)} blocked IPs from {self.blocklist_file}")
            else:
                # Create empty blocklist file
                self.save_blocklist()
                logging.info(f"Created new blocklist file: {self.blocklist_file}")
        except Exception as e:
            logging.error(f"Error loading blocklist: {e}")
            self.blocked_ips = set()
    
    def save_blocklist(self) -> None:
        """Save blocked IPs to JSON file"""
        try:
            data = {
                "blocked_ips": list(self.blocked_ips),
                "last_updated": datetime.now().isoformat(),
                "total_blocked": len(self.blocked_ips),
                "config": {
                    "rate_limit": self.rate_limit,
                    "time_window": self.time_window
                }
            }
            with open(self.blocklist_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving blocklist: {e}")
    
    @staticmethod
    def get_client_ip(request: Request) -> str:
        """Extract client IP from request with proxy support"""
        # Check for X-Forwarded-For (common in proxies)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
            if ip:
                return ip
        
        # Check for X-Real-IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to client host
        if request.client and request.client.host:
            return request.client.host
        
        return "unknown"
    
    def check_and_block(self, ip: str) -> Tuple[bool, str]:
        """
        Check if IP should be blocked and update tracking.
        
        Returns:
            Tuple[is_blocked, message]
        """
        current_time = time.time()
        
        # Check if already permanently blocked
        if ip in self.blocked_ips:
            return True, f"IP {ip} is permanently blocked"
        
        # Clean old requests from this IP
        valid_timestamps = [
            ts for ts in self.request_logs.get(ip, [])
            if current_time - ts < self.time_window
        ]
        self.request_logs[ip] = valid_timestamps
        
        # Check if exceeds rate limit
        if len(self.request_logs[ip]) >= self.rate_limit:
            # Block permanently
            self.blocked_ips.add(ip)
            self.save_blocklist()
            
            logging.warning(
                f"IP {ip} permanently blocked: {self.rate_limit} requests in {self.time_window} seconds"
            )
            message2 = f"IP {ip} permanently blocked: {self.rate_limit} requests in {self.time_window} seconds"
            SPCMailer.send_notification_email_sync( to="divesha@spc.int",
            subject="Ocean plotter - Blocklist",
            body=message2)
            return True, f"Rate limit exceeded ({self.rate_limit}/{self.time_window}s). IP {ip} permanently blocked."
        
        # Log the current request
        self.request_logs[ip].append(current_time)
        return False, ""
    
    def cleanup_old_logs(self) -> None:
        """Clean up old request logs to prevent memory leaks"""
        current_time = time.time()
        cutoff_time = current_time - (self.time_window * 2)  # Keep logs for 2x window
        
        for ip in list(self.request_logs.keys()):
            self.request_logs[ip] = [
                ts for ts in self.request_logs[ip]
                if ts > cutoff_time
            ]
            if not self.request_logs[ip]:
                del self.request_logs[ip]
    
    # Admin methods
    def get_blocklist(self) -> List[str]:
        """Get list of blocked IPs"""
        return list(self.blocked_ips)
    
    def block_ip(self, ip: str) -> bool:
        """Manually block an IP"""
        if ip not in self.blocked_ips:
            self.blocked_ips.add(ip)
            self.save_blocklist()
            return True
        return False
    
    def unblock_ip(self, ip: str) -> bool:
        """Manually unblock an IP"""
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            self.save_blocklist()
            return True
        return False
    
    def clear_blocklist(self) -> None:
        """Clear all blocked IPs"""
        self.blocked_ips.clear()
        self.save_blocklist()
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics"""
        return {
            "blocked_ips_count": len(self.blocked_ips),
            "active_tracked_ips": len(self.request_logs),
            "rate_limit": self.rate_limit,
            "time_window": self.time_window,
            "blocklist_file": str(self.blocklist_file)
        }