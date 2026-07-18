"""Hard-negative catalogues for finance-world stress cases."""

from __future__ import annotations

from distillery.data.world import CashRegime, TxnHardNegative, VarianceRegime

TRANSACTION_HARD_NEGATIVES: frozenset[TxnHardNegative] = frozenset(
    {
        TxnHardNegative.NEAR_SYNONYM_GL,
        TxnHardNegative.REFUND,
        TxnHardNegative.CAPEX_OPEX,
        TxnHardNegative.SPLIT_ALLOCATION,
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
        CashRegime.PARTIAL,
    }
)
