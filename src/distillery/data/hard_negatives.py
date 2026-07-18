"""Hard-negative catalogues for finance-world stress cases."""

from __future__ import annotations

from distillery.data.world import CashRegime, TxnHardNegative, VarianceRegime

TRANSACTION_HARD_NEGATIVES: frozenset[TxnHardNegative] = frozenset(
    {
        TxnHardNegative.NEAR_SYNONYM_GL,
        TxnHardNegative.REFUND,
        TxnHardNegative.CHARGEBACK,
        TxnHardNegative.CAPEX_OPEX,
        TxnHardNegative.SPLIT_ALLOCATION,
        TxnHardNegative.REFUND_SPLIT,
        TxnHardNegative.PERSONAL_LOOKING_ALLOWED,
        TxnHardNegative.ALLOWED_LOOKING_PROHIBITED,
        TxnHardNegative.CONFLICTING_RULES,
        TxnHardNegative.THRESHOLD_BOUNDARY,
        TxnHardNegative.MISLEADING_DESCRIPTOR,
    }
)

VARIANCE_HARD_REGIMES: frozenset[VarianceRegime] = frozenset(
    {
        VarianceRegime.OFFSET,
        VarianceRegime.PRICE_VOLUME,
        VarianceRegime.FX,
        VarianceRegime.TIE,
        VarianceRegime.HIDDEN_SUBTOTAL,
    }
)

CASH_HARD_REGIMES: frozenset[CashRegime] = frozenset(
    {
        CashRegime.BANK_FEE,
        CashRegime.DEPOSIT_IN_TRANSIT,
        CashRegime.STALE_CHECK,
        CashRegime.DUPLICATE,
        CashRegime.ONE_TO_MANY,
        CashRegime.MANY_TO_ONE,
        CashRegime.PARTIAL,
        CashRegime.SAME_AMOUNT_COLLISION,
    }
)
