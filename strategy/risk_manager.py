"""
Risk Manager
ICT Notes: Risk Management & Psychology.md, Passing Funded Accounts.md
SMM591: Trading Game — £10,000 demo account

Risk tiers (confidence-based):
  Confidence 4-5, no active losing streak : 1.0%  (high quality setup)
  All other cases                          : 0.5%  (standard)

Loss ladder (overrides tier):
  After 1+ consecutive loss  : max 0.5%
  After 3+ consecutive losses: max 0.25%
  After 5+ consecutive losses: block all entries until next day

Daily limits:
  Daily loss limit : 2% of equity  (stop trading for the rest of the day)
  Max trades/day   : 4
"""

from dataclasses import dataclass


INITIAL_EQUITY   = 10_000.0
DEFAULT_RISK_PCT = 0.005     # 0.5% base risk


@dataclass
class RiskManager:
    equity: float = INITIAL_EQUITY
    risk_pct: float = DEFAULT_RISK_PCT
    consecutive_losses: int = 0
    daily_loss: float = 0.0
    daily_loss_limit: float = 0.02   # 2% daily cap
    trades_today: int = 0
    max_trades_per_day: int = 4

    def current_risk_pct(self, confidence: int = 1) -> float:
        """
        Confidence 4-5 with clean slate → 1%.
        Any active losing streak caps at 0.5% then 0.25%.
        5+ consecutive losses → 0 (block trades, handled by can_trade).
        """
        if self.consecutive_losses >= 3:
            return 0.0025   # 0.25%
        if self.consecutive_losses >= 1:
            return 0.005    # 0.5% — no upgrading during a streak
        # Clean slate: reward quality setups
        if confidence >= 4:
            return 0.01     # 1% for high-confidence signals
        return 0.005        # 0.5% default

    def risk_amount(self, confidence: int = 1) -> float:
        return self.equity * self.current_risk_pct(confidence)

    def position_size(self, entry: float, stop: float, confidence: int = 1) -> float:
        """
        Calculate position size in units.
        risk_amount / distance_to_stop = position_size
        """
        distance = abs(entry - stop)
        if distance == 0:
            return 0.0
        return self.risk_amount(confidence) / distance

    def reward_amount(self, entry: float, target: float, size: float) -> float:
        return abs(target - entry) * size

    def risk_reward(self, entry: float, stop: float, target: float) -> float:
        risk = abs(entry - stop)
        reward = abs(target - entry)
        if risk == 0:
            return 0.0
        return reward / risk

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading conditions allow a new trade."""
        if self.consecutive_losses >= 5:
            return False, "5 consecutive losses — no new trades until next day"
        if self.daily_loss / self.equity >= self.daily_loss_limit:
            return False, f"Daily loss limit reached ({self.daily_loss_limit*100:.0f}% of equity)"
        if self.trades_today >= self.max_trades_per_day:
            return False, f"Max trades per day ({self.max_trades_per_day}) reached"
        return True, "OK"

    def record_trade(self, pnl: float):
        """Update state after a closed trade."""
        self.equity += pnl
        self.trades_today += 1
        if pnl < 0:
            self.consecutive_losses += 1
            self.daily_loss += abs(pnl)
        else:
            self.consecutive_losses = 0

    def reset_day(self):
        self.daily_loss = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0  # 5-loss block lifts at next day's open

    def summary(self) -> dict:
        return {
            "equity": round(self.equity, 2),
            "risk_pct": f"{self.current_risk_pct()*100:.1f}%",
            "risk_amount": round(self.risk_amount(), 2),
            "consecutive_losses": self.consecutive_losses,
            "daily_loss": round(self.daily_loss, 2),
            "return_pct": round((self.equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100, 2),
        }
